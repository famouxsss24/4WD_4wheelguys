import numpy as np
import cv2
import ai_edge_litert.interpreter as tflite
import time
import multiprocessing
import sys
import collections
from ultralytics import YOLO
import mycamera
from gpiozero import PWMOutputDevice, DigitalOutputDevice, Buzzer

PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)  
BIN2 = DigitalOutputDevice(24)

BUZZER = Buzzer(12) 

def set_motors(left_speed, right_speed):
    left_dir = "backward" if left_speed < 0 else "forward"
    right_dir = "backward" if right_speed < 0 else "forward"
    
    AIN1.value, AIN2.value = (0, 1) if left_dir == "forward" else (1, 0)
    BIN1.value, BIN2.value = (0, 1) if right_dir == "forward" else (1, 0)
    
    PWMA.value = max(0.0, min(abs(left_speed), 1.0))
    PWMB.value = max(0.0, min(abs(right_speed), 1.0))

def stop():
    PWMA.value, PWMB.value = 0, 0
    BUZZER.off()

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

    model_path = "model/best_640_int8.tflite" 
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
    history = collections.deque(maxlen=k_frames)

    while running_event.is_set():
        if not frame_queue.empty():
            frame_to_process = frame_queue.get()
            
            img_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            input_h, input_w = input_details[0]['shape'][1], input_details[0]['shape'][2]
            img_resized = cv2.resize(img_rgb, (input_w, input_h))
            img_input = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.squeeze(output_data) 
            
            boxes = output_data[:4, :].T
            scores_matrix = output_data[4:, :].T
            
            scores = np.max(scores_matrix, axis=1)
            class_ids = np.argmax(scores_matrix, axis=1)
            
            best_class_name = "없음"
            max_area = 0
            best_confidence = 0.0
            
            mask = scores >= 0.5
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            filtered_class_ids = class_ids[mask]
            
            for i in range(len(filtered_scores)):
                w, h = filtered_boxes[i][2], filtered_boxes[i][3]
                area = w * h
                
                if area > max_area:
                    max_area = area
                    best_class_name = class_names[filtered_class_ids[i]]
                    best_confidence = filtered_scores[i]

            history.append(best_class_name)

            if best_class_name != "없음" and history.count(best_class_name) >= n_threshold:
                shared_data['latest_sign_text'] = f"{best_class_name} ({best_confidence*100:.1f}%)"
            else:
                shared_data['latest_sign_text'] = "없음"
        else:
            time.sleep(0.01)

def main():
    manager = multiprocessing.Manager()
    shared_data = manager.dict()
    shared_data['latest_sign_text'] = "없음"
    
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

    lane_model_path = "model/model_quant_0516.tflite"
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

    camera = mycamera.MyPiCamera(640, 480)
    
    p = multiprocessing.Process(target=sign_detection_process, args=(frame_queue, shared_data, running_event))
    p.start()

    current_command = 0
    stop_time_end = 0.0
    limit_time_end = 0.0
    buzzer_time_end = 0.0
    sign_cooldown_end = 0.0
    cmd_reset_time_end = float('inf')
    intersection_sign_count = 0
    intersection_signs = ["left", "right", "straight", "red", "green"]
    
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
            sign_class = latest_sign_text.split(" ")[0] 

            if (sign_class == "stop" and has_stop_triggered) or \
               (sign_class == "limit" and has_limit_triggered) or \
               (sign_class == "brr" and has_brr_triggered) or \
               (sign_class in ["finish", "final"] and has_final_triggered):
                sign_class = "없음"

            if sign_class != "없음":
                if current_time > sign_cooldown_end:
                    cooldown_time = 4.0 

                    if sign_class in intersection_signs:
                        if current_command == 0 and sign_class == "straight":
                            cooldown_time = 0
                        else:
                            intersection_sign_count += 1
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

                            if intersection_sign_count == 1:
                                cooldown_time = 4.0
                                cmd_reset_time_end = float('inf')
                            elif intersection_sign_count == 2:
                                cooldown_time = 7.0
                                cmd_reset_time_end = float('inf')
                            elif intersection_sign_count >= 3:
                                cooldown_time = 4.0
                                cmd_reset_time_end = current_time + 4.0

                    elif sign_class == "stop":
                        stop_time_end = current_time + 3.5 
                        cooldown_time = 5.0
                        has_stop_triggered = True
                    elif sign_class == "limit":
                        limit_time_end = current_time + 4.0 
                        has_limit_triggered = True
                    elif sign_class == "brr":
                        buzzer_time_end = current_time + 1.0 
                        has_brr_triggered = True
                    elif sign_class in ["finish", "final"]: 
                        final_time_end = current_time + 4.0
                        has_final_triggered = True
                        cooldown_time = 5.0

                    sign_cooldown_end = current_time + cooldown_time

            if current_command in [1, 2, 3] and current_time > cmd_reset_time_end:
                current_command = 0  
                intersection_sign_count = 0
                cmd_reset_time_end = float('inf') 
                
            if current_time > final_time_end:
                is_finished = True

            if is_finished:
                print("\n\n🏁 종료라인 인식 주행을 마칩니다.")
                break

            if current_time < stop_time_end:
                set_motors(0, 0)
                if (current_time - last_print_time) > 0.3:
                    sys.stdout.write(f"\r[대기] 표지판:{latest_sign_text} | 카운트:{intersection_sign_count}/3 | 남은시간:{stop_time_end - current_time:.1f}s\033[K")
                    sys.stdout.flush()
                    last_print_time = current_time
                continue

            if current_time < limit_time_end:
                speedSet = 0.3
                speed_str = "서행(0.3)"
            else:
                speedSet = 0.5
                speed_str = "정상(0.5)"

            if current_time < buzzer_time_end:
                BUZZER.on()
            else:
                BUZZER.off()

            image_rgb = frame[:, :, ::-1]
            height = image_rgb.shape[0]
            roi_image = image_rgb[int(height * 0.60):, :, :]
            roi_resized = cv2.resize(roi_image, (320, 96))
            roi_yuv = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2YUV)
            
            input_data = np.float32(roi_yuv) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            input_cmd_data = np.zeros((1, 4), dtype=np.float32)
            input_cmd_data[0, current_command] = 2

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
                    P = np.clip(raw_P, -1.8, 1.8)
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
                cmd_str = ["외곽", "좌회전", "우회전", "교차로직진"][current_command]
                reset_str = f"리셋예약({cmd_reset_time_end - current_time:.1f}s)" if cmd_reset_time_end != float('inf') else "일반차선"
                sys.stdout.write(
                    f"\r[{speed_str}] CIL:{cmd_str} | 조향:{dir_str} | "
                    f"raw:{raw_prediction:.3f} | ema:{smoothed_prediction:.3f} | kp:{raw_P:.3f} | final:{P:.3f} | "
                    f"latency:{inference_ms:.1f}ms | "
                    f"표지판:{latest_sign_text} | 카운트:{intersection_sign_count}/3 | {reset_str}\033[K"
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
