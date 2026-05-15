import numpy as np
import cv2
import ai_edge_litert.interpreter as tflite # 차선 인식 모델용
import time
import multiprocessing
import sys
import collections
import mycamera
from ultralytics import YOLO # ONNX 모델 로드용
from gpiozero import PWMOutputDevice, DigitalOutputDevice, TonalBuzzer

# 핀 설정
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)  
BIN2 = DigitalOutputDevice(24)

BUZZER = TonalBuzzer(12) 

def set_motors(left_speed, right_speed, left_dir="forward", right_dir="forward"):
    AIN1.value, AIN2.value = (0, 1) if left_dir == "forward" else (1, 0)
    BIN1.value, BIN2.value = (0, 1) if right_dir == "forward" else (1, 0)
    
    PWMA.value = max(0.0, min(left_speed, 1.0))
    PWMB.value = max(0.0, min(right_speed, 1.0))

def stop():
    PWMA.value, PWMB.value = 0, 0
    BUZZER.stop()

# ==========================================
# 표지판 인식 프로세스 (ONNX 버전)
# ==========================================
def sign_detection_process(frame_queue, shared_data, running_event):
    model_path = "model/best.onnx" # ONNX 모델 경로
    try:
        # ONNX 모델은 YOLO 클래스에서 직접 로드 가능합니다.
        model = YOLO(model_path, task="detect")
    except Exception as e:
        shared_data['latest_sign_text'] = f"모델 로드 실패: {e}"
        return

    k_frames = 5
    n_threshold = 3
    history = collections.deque(maxlen=k_frames)

    while running_event.is_set():
        if not frame_queue.empty():
            frame_to_process = frame_queue.get()
            
            # YOLO ONNX 추론 (BGR 이미지를 그대로 사용)
            results = model(frame_to_process, verbose=False)
            
            best_class_name = "없음"
            max_area = 0
            best_confidence = 0.0
            
            for box in results[0].boxes:
                confidence = float(box.conf[0])
                
                # 임계값 0.5 설정
                if confidence >= 0.5: 
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > max_area:
                        max_area = area
                        best_class_name = class_name
                        best_confidence = confidence

            history.append(best_class_name)

            if best_class_name != "없음" and history.count(best_class_name) >= n_threshold:
                shared_data['latest_sign_text'] = f"{best_class_name} ({best_confidence*100:.1f}%)"
            else:
                shared_data['latest_sign_text'] = "없음"
        else:
            time.sleep(0.01)

# ==========================================
# 메인 자율주행 루프
# ==========================================
def main():
    manager = multiprocessing.Manager()
    shared_data = manager.dict()
    shared_data['latest_sign_text'] = "없음"
    
    frame_queue = multiprocessing.Queue(maxsize=1)
    running_event = multiprocessing.Event()
    running_event.set()

    # 차선 인식 모델 (TFLite 사용)
    lane_model_path = "model/my_rc_car_cil_model_normal.tflite"
    interpreter = tflite.Interpreter(model_path=lane_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_idx, cmd_idx = None, None
    for detail in input_details:
        if len(detail['shape']) == 4:
            img_idx = detail['index']
        elif len(detail['shape']) == 2:
            cmd_idx = detail['index']

    camera = mycamera.MyPiCamera(640, 480)
    
    # 표지판 인식 프로세스 시작
    p = multiprocessing.Process(target=sign_detection_process, args=(frame_queue, shared_data, running_event))
    p.start()

    current_command = 0
    intersection_sign_count = 0 # 변수 초기화 추가
    stop_time_end = 0.0
    limit_time_end = 0.0
    buzzer_time_end = 0.0
    sign_cooldown_end = 0.0
    cmd_reset_time_end = float('inf')
    intersection_signs = ["left", "right", "straight", "red", "green"]
    
    is_finished = False
    smoothed_prediction = 0.0  
    last_print_time = 0

    print("==========================================================")
    print("🚀 ONNX 기반 비동기 자율주행 시작! (core_onnx.py)")
    print("==========================================================")
    time.sleep(1)

    try:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret: continue
            
            # 카메라 반전 및 복사(Contiguous 메모리 확보)
            frame = frame[::-1, ::-1, :].copy()
            current_time = time.time()

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except:
                    pass
            frame_queue.put(frame)

            latest_sign_text = shared_data['latest_sign_text']
            sign_class = latest_sign_text.split(" ")[0] 

            if sign_class != "없음":
                # 쿨다운(sign_cooldown_end)이 지났을 때만 인식 및 카운트 진행
                if current_time > sign_cooldown_end:
                    
                    if sign_class in intersection_signs:
                        # 1. 카운트 증가
                        intersection_sign_count += 1
                        
                        # 2. 중복 인식 방지 쿨다운 설정
                        if intersection_sign_count == 2:
                            # 2번째 인식 후 3번째 인식까지는 5초 대기 (탈출 감지 지연)
                            sign_cooldown_end = current_time + 5.0 
                        else:
                            # 그 외에는 3초 대기
                            sign_cooldown_end = current_time + 3.0 
                        
                        # 3. 첫 번째 인식 시 즉시 주행 모드 변경
                        if intersection_sign_count == 1:
                            if sign_class == "left":
                                current_command = 1
                            elif sign_class == "right":
                                current_command = 2
                            elif sign_class == "straight":
                                current_command = 3
                            elif sign_class == "red":
                                current_command = 1                  
                                stop_time_end = current_time + 3.0   
                            elif sign_class == "green":
                                current_command = 2                  
                            print(f"\n[!] 교차로 진입 (1/3): {sign_class} 모드로 변경")
                        
                        print(f"    -> 표지판 인식 카운트: {intersection_sign_count}/3")

                        # 4. 세 번째 인식 성공 시 3초 후 복귀 예약
                        if intersection_sign_count >= 3:
                            cmd_reset_time_end = current_time + 3.0
                            intersection_sign_count = 0 # 다음 교차로를 위해 초기화
                            print(f"[!] 교차로 탈출 감지 (3/3): 3초 후 C0 복귀")

                    elif sign_class == "stop":
                        stop_time_end = current_time + 3.5 
                        sign_cooldown_end = current_time + 5.0
                    elif sign_class == "limit":
                        limit_time_end = current_time + 3.0 
                        sign_cooldown_end = current_time + 5.0
                    elif sign_class == "brr":
                        buzzer_time_end = current_time + 1.0 
                        sign_cooldown_end = current_time + 3.0
                    elif sign_class == "final": 
                        is_finished = True

            # C0 복귀 로직
            if current_command in [1, 2, 3] and current_time > cmd_reset_time_end:
                print("\n[!] 교차로 통과 완료, C0(외곽) 모드로 복귀합니다.")
                current_command = 0  
                cmd_reset_time_end = float('inf') 
                
            if is_finished:
                print("\n\n🏁 종료라인 인식 주행을 마칩니다.")
                break

            if current_time < stop_time_end:
                set_motors(0, 0)
                continue

            # 속도 및 부저 제어
            speedSet = 0.3 if current_time < limit_time_end else 0.5
            if current_time < buzzer_time_end:
                if not BUZZER.is_active:
                    BUZZER.play(440.0)
            else:
                BUZZER.stop()

            # 차선 주행 추론 (TFLite)
            image_rgb = frame[:, :, ::-1]
            height = image_rgb.shape[0]
            roi_image = image_rgb[int(height * 0.60):, :, :]
            roi_resized = cv2.resize(roi_image, (320, 96))
            roi_yuv = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2YUV)
            
            input_data = np.float32(roi_yuv) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            input_cmd_data = np.zeros((1, 4), dtype=np.float32)
            input_cmd_data[0, current_command] = 1.0

            interpreter.set_tensor(img_idx, input_data)
            interpreter.set_tensor(cmd_idx, input_cmd_data)
            interpreter.invoke()
            raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            alpha = 0.8
            smoothed_prediction = (1 - alpha) * smoothed_prediction + alpha * raw_prediction
            
            sensitive = 0.08
            Kp = 2.5
            
            if abs(smoothed_prediction) <= sensitive:
                P = 0.0
            else:
                P = np.clip(smoothed_prediction * Kp, -1.0, 1.0)
            
            if P == 0.0:
                left_pwm, right_pwm = speedSet, speedSet
                dir_str = "⬆️ 직진"
            elif P > 0:
                left_pwm, right_pwm = speedSet, speedSet * (1.0 - P)
                dir_str = "➡️ 우조향"
            else:
                left_pwm, right_pwm = speedSet * (1.0 + P), speedSet
                dir_str = "⬅️ 좌조향"

            if (current_time - last_print_time) > 0.3:
                cmd_str = ["외곽", "좌회전", "우회전", "교차로직진"][current_command]
                # 카운트 정보 추가
                sys.stdout.write(f"\r[{speedSet}] CIL:{cmd_str:<5} | 조향:{dir_str:<5} | 카운트:{intersection_sign_count}/3 | 표지판:{latest_sign_text:<15}   ")
                sys.stdout.flush()
                last_print_time = current_time

            set_motors(left_pwm, right_pwm)
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        running_event.clear()
        print("\n\n모터를 정지하고 시스템을 종료합니다.")
        stop()
        p.join(timeout=1)
        if p.is_alive():
            p.terminate()
        if 'camera' in locals() and camera.isOpened():
            camera.release()

if __name__ == '__main__':
    main()
