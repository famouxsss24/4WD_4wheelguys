import numpy as np
import cv2
import ai_edge_litert.interpreter as tflite
import time
import threading
import sys
import collections # 💡 k프레임 히스토리 저장을 위해 추가
from ultralytics import YOLO
import mycamera
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Buzzer

# ==========================================
# 1. 핀 설정 및 하드웨어 제어
# ==========================================
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)  
BIN2 = DigitalOutputDevice(24)

# !!!!!부저 핀 설정 (환경에 맞게 GPIO 번호 확인)
BUZZER = Buzzer(17) 

def set_motors(left_speed, right_speed, left_dir="forward", right_dir="forward"):
    AIN1.value, AIN2.value = (0, 1) if left_dir == "forward" else (1, 0)
    BIN1.value, BIN2.value = (0, 1) if right_dir == "forward" else (1, 0)
    
    PWMA.value = max(0.0, min(left_speed, 1.0))
    PWMB.value = max(0.0, min(right_speed, 1.0))

def stop():
    PWMA.value, PWMB.value = 0, 0
    BUZZER.off()

# ==========================================
# 2. 비동기 표지판 인식을 위한 전역 변수
# ==========================================
latest_frame = None
latest_sign_text = "없음"
running = True

# ==========================================
# 3. 비동기 표지판 인식 스레드 함수 (YOLO)
# ==========================================
def sign_detection_thread():
    global latest_frame, latest_sign_text, running
    
    model_path = "code/model/best_0505.onnx" 
    try:
        model = YOLO(model_path)
    except Exception as e:
        latest_sign_text = "모델 로드 실패"
        return

    # 💡 [핵심] k 프레임 중 n 번 인식 설정 !!!!!!!!!표지판을 늦게 인식한다면 3,2 로 변경
    k_frames = 5
    n_threshold = 3
    history = collections.deque(maxlen=k_frames) # 최근 5프레임의 결과를 담는 큐

    while running:
        if latest_frame is not None:
            frame_to_process = latest_frame.copy()
            results = model(frame_to_process, verbose=False) #!!!!!YOLO가 표지판을 너무 늦게 인식해서 교차로 진입 타이밍을 놓친다면 results = model(frame_to_process, imgsz=320, verbose=False)
            
            best_class_name = "없음"
            max_area = 0
            best_confidence = 0.0
            
            for box in results[0].boxes:
                confidence = float(box.conf[0])
                
                # 💡 조건 1: 신뢰도 70% 이상
                if confidence >= 0.7: 
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # 💡 조건 2: 바운딩 박스 면적 계산 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    
                    # 화면에 여러 표지판이 있으면 면적이 가장 큰(가장 가까운) 것을 선택
                    if area > max_area:
                        max_area = area
                        best_class_name = class_name
                        best_confidence = confidence

            # 현재 프레임에서 최종 산출된 결과를 히스토리에 추가
            history.append(best_class_name)

            # 💡 조건 3: 최근 k프레임 중 n번 이상 동일한 객체가 잡혔을 때만 확정
            if best_class_name != "없음" and history.count(best_class_name) >= n_threshold:
                latest_sign_text = f"{best_class_name} ({best_confidence*100:.1f}%)"
            else:
                # 조건을 만족하지 못하면 메인 루프에 이벤트를 주지 않음
                latest_sign_text = "없음"
        
        time.sleep(0.05)

# ==========================================
# 4. 메인 루프 (자율주행 및 상태 머신)
# ==========================================
def main():
    global latest_frame, running

    # --- CIL(차선 인식) 모델 로드 ---
    lane_model_path = "code/model/my_rc_car_cil_model_normal.tflite"
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
    
    detect_thread = threading.Thread(target=sign_detection_thread)
    detect_thread.daemon = True
    detect_thread.start()

    # --- 상태 제어용 변수 ---
    current_command = 0        # 0:외곽, 1:좌회전, 2:우회전, 3:직진
    stop_time_end = 0.0        # 일시정지 종료 시간
    limit_time_end = 0.0       # 속도제한 종료 시간
    buzzer_time_end = 0.0      # 부저 종료 시간
    sign_cooldown_end = 0.0    # 중복 인식 방지 쿨다운 시간
    cmd_reset_time_end = float('inf') # 복귀 타이머 (초기값 무한대)
    intersection_sign_count = 0       # 교차로 표지판 카운터
    intersection_signs = ["left", "right", "straight", "red", "green"] # 카운트 대상
    
    is_finished = False
    smoothed_prediction = 0.0  
    last_print_time = 0

    print("==========================================================")
    print("🚀 완전 자율주행(정밀 표지판 필터링 적용) 시작! (Ctrl+C 종료)")
    print("==========================================================")
    time.sleep(1)

    try:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret: continue
            
            frame = frame[::-1, ::-1, :]
            latest_frame = frame
            current_time = time.time() #sleep을 쓰면 n초 동안 파이썬 프로그램 전체가 멈춰버림. * 카메라 영상을 새로 읽어오지 못함

            # ==========================================
            # [핵심 로직 1] 표지판 인식 및 카운터 로직
            # ==========================================
            sign_class = latest_sign_text.split(" ")[0] 

            if sign_class != "없음":
                # 💡 조건 4: 쿨다운 적용 (같은 표지판 연속 인식 방지)
                if current_time > sign_cooldown_end:
                    
                    cooldown_time = 2.5 # 무한 정지 방지를 위한 기본 쿨다운 설정

                    # 교차로 관련 표지판인 경우 (5개 중 하나)
                    if sign_class in intersection_signs:
                        
                        # 💡 [핵심 방어 로직] 외곽(C0) 주행 중 'straight'를 보면 완전히 무시!
                        if current_command == 0 and sign_class == "straight":
                            cooldown_time = 0 # 💡 추가된 수정: 무시했으므로 쿨다운도 먹이지 않음 (유령 쿨다운 방지)
                        
                        else:
                            # 위 조건을 통과한 경우에만 카운터 증가
                            intersection_sign_count += 1
                            
                            # 각 표지판에 따른 액션 부여
                            if sign_class == "left":
                                current_command = 1
                            elif sign_class == "right":
                                current_command = 2
                            elif sign_class == "straight":
                                current_command = 3 # (외곽이 아니므로) 정상적으로 C3으로 전환
                            elif sign_class == "red":
                                current_command = 1                  
                                stop_time_end = current_time + 3.0   
                                cooldown_time = 4.5 # 무한정지 방지
                            elif sign_class == "green":
                                current_command = 2                  

                            # 카운트가 3이 되면 탈출 타이머 가동
                            if intersection_sign_count >= 3:
                                cmd_reset_time_end = current_time + 3.0  #!!!실제 트랙 상황에 맞게 조절 필요
                                intersection_sign_count = 0 
                            else:
                                cmd_reset_time_end = float('inf')

                    # 기타 일반 표지판
                    elif sign_class == "stop":
                        stop_time_end = current_time + 3.5 
                        cooldown_time = 5.0 # 무한정지 방지
                    elif sign_class == "limit":
                        limit_time_end = current_time + 4.0 
                    elif sign_class == "brr":
                        buzzer_time_end = current_time + 1.0 
                    elif sign_class == "finish": 
                        is_finished = True

                    # 쿨다운 타이머 갱신 (트랙에 맞춰 조절)
                    sign_cooldown_end = current_time + cooldown_time  #!!!실제 트랙 상황에 맞게 조절 필요

            # ==========================================
            # [핵심 로직 2] 교차로 탈출 완료 감지
            # ==========================================
            if current_command in [1, 2, 3] and current_time > cmd_reset_time_end:
                current_command = 0  
                cmd_reset_time_end = float('inf') 
                
            # ==========================================
            # [차량 제어 로직] 상태에 따른 액션 수행
            # ==========================================
            
            # 1. 평가 종료 조건
            if is_finished:
                print("\n\n🏁 종료라인 인식 주행을 마칩니다.")
                break

            # 2. 정지 상태 
            if current_time < stop_time_end:
                set_motors(0, 0)
                sys.stdout.write(f"\r[🛑 대기 중] 표지판: {latest_sign_text:<15} 카운트: {intersection_sign_count}/3   ")
                sys.stdout.flush()
                continue

            # 3. 속도 설정 
            if current_time < limit_time_end:
                speedSet = 0.3
                speed_str = "⚠️ 서행(0.3)"
            else:
                speedSet = 0.5
                speed_str = "▶️ 정상(0.5)"

            # 4. 부저 제어 
            if current_time < buzzer_time_end:
                BUZZER.on()
            else:
                BUZZER.off()

            # ==========================================
            # [차선 주행 로직] TFLite CIL 모델 추론
            # ==========================================
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

            # 콘솔 상태 출력 (입력 지연 방지를 위해 0.3초마다 갱신)
            if (current_time - last_print_time) > 0.3:
                cmd_str = ["외곽", "좌회전", "우회전", "교차로직진"][current_command]
                sys.stdout.write(f"\r[{speed_str}] CIL:{cmd_str:<5} | 조향:{dir_str:<5} | 표지판:{latest_sign_text:<15}   ")
                sys.stdout.flush()
                last_print_time = current_time

            set_motors(left_pwm, right_pwm)
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        running = False
        print("\n\n모터를 정지하고 시스템을 종료합니다.")
        stop()
        if 'camera' in locals() and camera.isOpened():
            camera.release()

if __name__ == '__main__':
    main()