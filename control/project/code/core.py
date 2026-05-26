import numpy as np
import cv2
import ai_edge_litert.interpreter as tflite
import time
import multiprocessing
import sys
import collections
from ultralytics import YOLO
import mycamera
from gpiozero import PWMOutputDevice, DigitalOutputDevice, TonalBuzzer

PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)  
BIN2 = DigitalOutputDevice(24)

BUZZER = TonalBuzzer(12) 

SIGN_MODEL_PATH = "model/best_320_int8.tflite"
LANE_MODEL_PATH = "model/model_quant_0523.tflite"

def set_motors(left_speed, right_speed):
    left_dir = "backward" if left_speed < 0 else "forward"
    right_dir = "backward" if right_speed < 0 else "forward"
    
    AIN1.value, AIN2.value = (0, 1) if left_dir == "forward" else (1, 0)
    BIN1.value, BIN2.value = (0, 1) if right_dir == "forward" else (1, 0)
    
    PWMA.value = max(0.0, min(abs(left_speed), 1.0))
    PWMB.value = max(0.0, min(abs(right_speed), 1.0))

def stop():
    PWMA.value, PWMB.value = 0, 0
    try:
        BUZZER.stop()
    except Exception:
        pass

def sign_detection_process(frame_queue, shared_data, running_event):
    import os
    try:
        cpu_count = os.cpu_count() or 4
        if cpu_count >= 4:
            os.sched_setaffinity(0, {2, 3})
        elif cpu_count >= 2:
            os.sched_setaffinity(0, {1})
    except Exception:
        pass

    model_path = SIGN_MODEL_PATH 
    try:
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=2)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        class_names = ['brr', 'green', 'left', 'limit', 'red', 'right', 'stop', 'straight','final']
    except Exception as e:
        shared_data['latest_sign_text'] = f"모델 로드 실패: {e}"
        return

    k_frames = 5
    n_threshold = 3
    histories = {cls: collections.deque(maxlen=k_frames) for cls in class_names}

    while running_event.is_set():
        if not frame_queue.empty():
            frame_to_process = frame_queue.get()
            
            img_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            input_h, input_w = input_details[0]['shape'][1], input_details[0]['shape'][2]
            
            if img_rgb.shape[1] == input_w and img_rgb.shape[0] == input_h:
                img_resized = img_rgb
            else:
                img_resized = cv2.resize(img_rgb, (input_w, input_h))
                
            img_input = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], img_input)
            t_yolo0 = time.time()
            interpreter.invoke()
            shared_data['yolo_latency_ms'] = (time.time() - t_yolo0) * 1000
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.squeeze(output_data) 
            
            boxes = output_data[:4, :].T
            scores_matrix = output_data[4:, :].T
            
            scores = np.max(scores_matrix, axis=1)
            class_ids = np.argmax(scores_matrix, axis=1)
            
            current_frame_detections = {}
            
            mask = scores >= 0.65
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            filtered_class_ids = class_ids[mask]
            
            for i in range(len(filtered_scores)):
                w, h = filtered_boxes[i][2], filtered_boxes[i][3]
                area = w * h
                cls_name = class_names[filtered_class_ids[i]]
                conf = filtered_scores[i]
                
                if cls_name not in current_frame_detections or area > current_frame_detections[cls_name]['area']:
                    current_frame_detections[cls_name] = {'area': area, 'conf': conf}
            
            for cls in class_names:
                if cls in current_frame_detections:
                    histories[cls].append((current_frame_detections[cls]['area'], current_frame_detections[cls]['conf']))
                else:
                    histories[cls].append(None)
            
            valid_candidates = []
            for cls in class_names:
                valid_samples = [x for x in histories[cls] if x is not None]
                if len(valid_samples) >= n_threshold:
                    latest_area, latest_conf = valid_samples[-1]
                    if latest_area >= 0.03:
                        valid_candidates.append((cls, latest_area, latest_conf))
            
            if len(valid_candidates) > 0:
                valid_candidates.sort(key=lambda x: x[1], reverse=True)
                best_cls, best_area, best_conf = valid_candidates[0]
                shared_data['latest_sign_text'] = f"{best_cls} ({best_conf*100:.1f}%)"
            else:
                shared_data['latest_sign_text'] = "없음"
        else:
            time.sleep(0.01)

def main():
    manager = multiprocessing.Manager()
    shared_data = manager.dict()
    shared_data['latest_sign_text'] = "없음"
    shared_data['yolo_latency_ms'] = 0.0
    
    frame_queue = multiprocessing.Queue(maxsize=1)
    running_event = multiprocessing.Event()
    running_event.set()

    import os
    try:
        cpu_count = os.cpu_count() or 4
        if cpu_count >= 4:
            os.sched_setaffinity(0, {0, 1})
        elif cpu_count >= 2:
            os.sched_setaffinity(0, {0})
    except Exception:
        pass

    lane_model_path = LANE_MODEL_PATH
    interpreter = tflite.Interpreter(model_path=lane_model_path, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_idx, cmd_idx = None, None
    for detail in input_details:
        if len(detail['shape']) == 4:
            img_idx = detail['index']
        elif len(detail['shape']) == 2:
            cmd_idx = detail['index']

    camera = mycamera.MyPiCamera(320, 240)
    
    p = multiprocessing.Process(target=sign_detection_process, args=(frame_queue, shared_data, running_event))
    p.start()

    current_command = 0
    stop_time_end = 0.0
    limit_time_end = 0.0
    buzzer_time_end = 0.0
    sign_cooldown_end = 0.0
    cmd_reset_time_end = float('inf')
    intersection_signs = ["left", "right", "straight", "red", "green"]
    
    ZONE_OUTER = 0
    ZONE_INNER = 1
    ZONE_EXIT = 2
    ZONE_RECOVERY = 3
    current_zone = ZONE_OUTER
    sign_locked = False
    recovery_time_end = 0.0
    zone_transition_time = 0.0
    
    # FSM 데바운스 및 홀드 필터 파라미터
    DEBOUNCE_TIME = 2.0
    HOLD_TIME = 1.0
    SIGN_COOLDOWN_TIME = 0.5
    hold_sign = "없음"
    hold_start_time = 0.0
    
    has_stop_triggered = False
    has_limit_triggered = False
    has_brr_triggered = False
    has_final_triggered = False
    final_time_end = float('inf')
    last_inference_time = time.time()
    inference_ms = 0.0
    
    is_finished = False
    smoothed_prediction = 0.0  
    last_print_time = 0

    print("==========================================================")
    print("🚀 Multiprocessing 기반 자율주행 시작! (core.py)")
    print("==========================================================")
    time.sleep(1)

    try:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret: continue
            
            frame = frame[::-1, ::-1, :]
            current_time = time.time()


            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except:
                    pass
            frame_queue.put(frame)

            latest_sign_text = shared_data['latest_sign_text']
            raw_sign = latest_sign_text.split(" ")[0] 

            if (raw_sign == "stop" and has_stop_triggered) or \
               (raw_sign == "limit" and has_limit_triggered) or \
               (raw_sign == "brr" and has_brr_triggered) or \
               (raw_sign in ["finish", "final"] and has_final_triggered):
                raw_sign = "없음"

            # 신호 홀드 필터 (1-0-1 -> 1)
            if raw_sign != "없음":
                hold_sign = raw_sign
                hold_start_time = current_time
            else:
                if current_time - hold_start_time > HOLD_TIME:
                    hold_sign = "없음"

            sign_class = hold_sign

            # Falling Edge 감지
            if sign_class == "없음":
                if sign_locked:
                    # 최소 구역 유지 시간(Debounce Time) 검사
                    if current_time - zone_transition_time > DEBOUNCE_TIME:
                        sign_locked = False
                        sign_cooldown_end = current_time + SIGN_COOLDOWN_TIME
                        if current_zone == ZONE_EXIT:
                            # 3단계 (ZONE_EXIT)에서 표지판이 사라진 순간(Falling Edge) 복귀 단계 진입
                            current_zone = ZONE_RECOVERY
                            recovery_time_end = current_time + 5.0
                            zone_transition_time = current_time
                            print(f"\n[FSM] ZONE_EXIT -> ZONE_RECOVERY: 표지판 사라짐 감지 (5초 뒤 외곽 복귀 예약)")

            # Rising Edge 감지 (유예 시간 경과 시 활성화)
            if sign_class != "없음":
                if not sign_locked and (current_time > sign_cooldown_end):
                    if sign_class in intersection_signs:
                        # 구역별 표지판 타당성(Gating) 검사
                        is_valid_sign = True
                        if current_zone == ZONE_OUTER and sign_class not in ["left", "right"]:
                            is_valid_sign = False
                        elif current_zone == ZONE_EXIT and sign_class not in ["left", "right"]:
                            is_valid_sign = False

                        if is_valid_sign:
                            sign_locked = True  # Rising Edge 진입으로 인한 잠금(Lock) 활성화
                            zone_transition_time = current_time  # 데바운스 기준 시간 갱신
                            
                            if current_zone == ZONE_OUTER:
                                # 1단계 (ZONE_OUTER): 교차로 진입 대기 및 지령 설정
                                if sign_class == "left":
                                    current_command = 1
                                elif sign_class == "right":
                                    current_command = 2
                                
                                current_zone = ZONE_INNER
                                print(f"\n[FSM] ZONE_OUTER -> ZONE_INNER: {sign_class} 감지 (cmd: {current_command})")

                            elif current_zone == ZONE_INNER:
                                # 2단계 (ZONE_INNER): 교차로 내부 주행 중 내부 액션 지령 수신
                                if sign_class == "left":
                                    current_command = 1
                                elif sign_class == "right":
                                    current_command = 2
                                elif sign_class == "straight":
                                    current_command = 3
                                elif sign_class == "red":
                                    current_command = 2                  
                                    stop_time_end = current_time + 3.0   
                                elif sign_class == "green":
                                    current_command = 2
                                
                                current_zone = ZONE_EXIT
                                print(f"\n[FSM] ZONE_INNER -> ZONE_EXIT: {sign_class} 감지 (cmd: {current_command})")

                            elif current_zone == ZONE_EXIT:
                                # 3단계 (ZONE_EXIT): 탈출 명령 표지판이 들어왔을 때 최종 탈출 명령 주입 (Rising Edge)
                                if sign_class == "left":
                                    current_command = 1
                                elif sign_class == "right":
                                    current_command = 2
                                print(f"\n[FSM] ZONE_EXIT: {sign_class} 감지 (탈출 cmd: {current_command} 적용)")
                    else:
                        # 일반 액션 표지판 (stop, limit, brr, finish/final)
                        sign_locked = True
                        zone_transition_time = current_time
                        
                        if sign_class == "stop":
                            stop_time_end = current_time + 3.5 
                            has_stop_triggered = True
                            print(f"\n[SIGN] stop 감지 (3.5초 대기)")
                        elif sign_class == "limit":
                            limit_time_end = current_time + 4.0 
                            has_limit_triggered = True
                            print(f"\n[SIGN] limit 감지 (4초 서행)")
                        elif sign_class == "brr":
                            buzzer_time_end = current_time + 1.0 
                            has_brr_triggered = True
                            try:
                                BUZZER.play(391)
                            except Exception:
                                pass
                            print(f"\n[SIGN] brr 감지 (1초 버저)")
                        elif sign_class in ["finish", "final"]: 
                            final_time_end = current_time + 4.0
                            has_final_triggered = True
                            print(f"\n[SIGN] finish/final 감지 (4초 뒤 종료)")

            # 복귀 단계 (ZONE_RECOVERY) 처리: 타이머 만료 시 외곽으로 복귀
            if current_zone == ZONE_RECOVERY:
                if current_time > recovery_time_end:
                    current_command = 0
                    current_zone = ZONE_OUTER
                    print(f"\n[FSM] ZONE_RECOVERY -> ZONE_OUTER: 외곽 차선 복귀 완료 (cmd: 0)")


            if current_time > final_time_end:
                is_finished = True

            if is_finished:
                print("\n\n🏁 종료라인 인식 주행을 마칩니다.")
                break

            if current_time < stop_time_end:
                set_motors(0, 0)
                if (current_time - last_print_time) > 0.3:
                    zone_str = ["외곽", "내부선회", "탈출대기", "복귀주행"][current_zone]
                    sys.stdout.write(f"\r[대기] 표지판:{latest_sign_text} | 구역:{zone_str} | 남은시간:{stop_time_end - current_time:.1f}s\033[K")
                    sys.stdout.flush()
                    last_print_time = current_time
                continue

            if current_time < limit_time_end:
                speedSet = 0.2
                speed_str = "서행"
            else:
                speedSet = 0.5
                speed_str = "정상"

            if buzzer_time_end > 0.0 and current_time >= buzzer_time_end:
                try:
                    BUZZER.stop()
                except Exception:
                    pass
                buzzer_time_end = 0.0

            image_rgb = frame[:, :, ::-1]
            height = image_rgb.shape[0]
            roi_image = image_rgb[int(height * 0.60):, :, :]
            roi_yuv = cv2.cvtColor(roi_image, cv2.COLOR_RGB2YUV)
            
            input_data = np.float32(roi_yuv) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            input_cmd_data = np.zeros((1, 4), dtype=np.float32)
            input_cmd_data[0, current_command] = 1.0

            interpreter.set_tensor(img_idx, input_data)
            interpreter.set_tensor(cmd_idx, input_cmd_data)
            
            t0 = time.time()
            interpreter.invoke()
            inference_ms = (time.time() - t0) * 1000
            
            raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            last_inference_time = time.time()
            
            if current_command in [1, 2]:
                smoothed_prediction = raw_prediction
            else:
                alpha = 0.95
                smoothed_prediction = (1 - alpha) * smoothed_prediction + alpha * raw_prediction
            
            sensitive = 0.08
            Kp = 2.5
            raw_P = smoothed_prediction * Kp
            
            if abs(smoothed_prediction) <= sensitive:
                P = 0.0
            else:
                if current_command in [1, 2]:
                    P = np.clip(raw_P, -2.0, 2.0)
                else:
                    P = np.clip(raw_P, -1.0, 1.0)
            
            if P == 0.0:
                left_pwm, right_pwm = speedSet, speedSet
                dir_str = "직진"
            elif P > 0:
                left_pwm, right_pwm = speedSet, speedSet * (1.0 - P)
                dir_str = "우조향"
            else:
                left_pwm, right_pwm = speedSet * (1.0 + P), speedSet
                dir_str = "좌조향"

            if (current_time - last_print_time) > 0.3:
                zone_str = ["외곽", "내부선회", "탈출대기", "복귀주행"][current_zone]
                yolo_latency_ms = shared_data.get('yolo_latency_ms', 0.0)
                sys.stdout.write(
                    f"\rC[{current_command}] | raw:{raw_prediction:.3f} | "
                    f"P:{P:.3f} | "
                    f"구역:{zone_str} | 표지판:{sign_class} | "
                    f"L : {inference_ms:.1f}ms , Y : {yolo_latency_ms:.1f}ms\033[K"
                )
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
