import numpy as np
import cv2
import ai_edge_litert.interpreter as tflite
import time
import threading
import sys
import select
import tty
import termios
from ultralytics import YOLO
import mycamera
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# ==========================================
# 1. 모터 초기화
# ==========================================
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)  
BIN2 = DigitalOutputDevice(24)

def set_motors(left_speed, right_speed, left_dir="forward", right_dir="forward"):
    AIN1.value, AIN2.value = (0, 1) if left_dir == "forward" else (1, 0)
    BIN1.value, BIN2.value = (0, 1) if right_dir == "forward" else (1, 0)
    
    PWMA.value = max(0.0, min(left_speed, 1.0))
    PWMB.value = max(0.0, min(right_speed, 1.0))

def stop():
    PWMA.value, PWMB.value = 0, 0

# ==========================================
# 2. 비동기 표지판 인식을 위한 전역 변수 및 CIL 커맨드
# ==========================================
latest_frame = None
latest_sign_text = "없음"
running = True
current_command = 0  # 초기값: C0 (외곽 주행)

# ==========================================
# 3. 비동기 표지판 인식 스레드 함수
# ==========================================
def sign_detection_thread():
    global latest_frame, latest_sign_text, running
    
    model_path = "code/model/best_0505.onnx"
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        latest_sign_text = "모델 로드 실패"
        return

    while running:
        if latest_frame is not None:
            # 원본 프레임 복사 (스레드 충돌 방지)
            frame_to_process = latest_frame.copy()
            
            # YOLO 모듈은 OpenCV 표준인 BGR을 기대하므로 그대로 사용
            results = model(frame_to_process, verbose=False)
            
            detected = False
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # 신뢰도가 50% 이상일 때만 표시
                if confidence > 0.5:
                    class_name = model.names[class_id]
                    latest_sign_text = f"{class_name} ({confidence*100:.1f}%)"
                    detected = True
                    break # 첫 번째로 인식된 (가장 확신하는) 표지판만 출력
            
            if not detected:
                latest_sign_text = "없음"
        
        # CPU 100% 사용 방지
        time.sleep(0.05)

# ==========================================
# 4. 실시간 CIL 커맨드 입력 스레드
# ==========================================
def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def command_listener():
    global current_command, running
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        # 입력된 키가 터미널에 에코(출력)되지 않도록 설정하여 시각적 충돌 방지
        new_settings = termios.tcgetattr(sys.stdin)
        new_settings[3] = new_settings[3] & ~termios.ECHO
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)
        
        while running:
            if isData():
                key = sys.stdin.read(1)
                if key == '1':
                    current_command = 0
                elif key == '2':
                    current_command = 1
                elif key == '3':
                    current_command = 2
                elif key == '4':
                    current_command = 3
                elif key == 'k' or key == 'K':
                    running = False
                    break
            time.sleep(0.1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

# ==========================================
# 5. 메인 (차선 인식 및 주행)
# ==========================================
def main():
    global latest_frame, running, current_command

    # 차선 인식 모델 로드 (CIL 모델로 변경)
    lane_model_path = "code/model/my_rc_car_cil_model_normal.tflite"
    interpreter = tflite.Interpreter(model_path=lane_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 텐서 인덱스 동적 할당
    img_idx = None
    cmd_idx = None
    for detail in input_details:
        if len(detail['shape']) == 4:
            img_idx = detail['index']
        elif len(detail['shape']) == 2:
            cmd_idx = detail['index']

    # 카메라 셋팅 (640x480으로 통합)
    camera = mycamera.MyPiCamera(640, 480)
    
    # 표지판 인식 백그라운드 스레드 시작
    detect_thread = threading.Thread(target=sign_detection_thread)
    detect_thread.daemon = True
    detect_thread.start()

    # 커맨드 입력 스레드 시작
    cmd_thread = threading.Thread(target=command_listener)
    cmd_thread.daemon = True
    cmd_thread.start()

    speedSet = 0.5
    smoothed_prediction = 0.0  
    last_print_time = 0
    last_cmd = -1
    last_sign = ""

    print("==========================================================")
    print("🚀 자율주행(CIL) 및 비동기 표지판 인식 통합 시스템 시작! (K 종료)")
    print(" [ 1:외곽 | 2:좌회전 | 3:우회전 | 4:직진 | K:종료 ] ")
    print("==========================================================")
    
    # 카메라 웜업 대기
    time.sleep(1)

    try:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                continue
            
            # ⚠️ 카메라 물리적 상하좌우 반전
            frame = frame[::-1, ::-1, :]
            
            # 표지판 인식 스레드에서 접근할 수 있도록 글로벌 변수 업데이트 (BGR 형식)
            latest_frame = frame
            
            # --- 차선 인식 (Lane Tracking) 파이프라인 ---
            # 1. BGR에서 RGB로 변환
            image_rgb = frame[:, :, ::-1]
            
            # 2. 아래 40% 자르기 (높이 480의 0.6 = 288 인덱스부터) -> 640x192 이미지
            height = image_rgb.shape[0]
            roi_image = image_rgb[int(height * 0.60):, :, :]
            
            # 3. 크기 변환: 640x192 -> 320x96
            # 320x240 해상도 시절의 40%(320x96)와 동일한 입력을 모델에 제공하기 위함
            roi_resized = cv2.resize(roi_image, (320, 96))
            
            # 4. YUV로 변환
            roi_yuv = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2YUV)
            
            input_data = np.float32(roi_yuv) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            # CIL 커맨드 전처리 (원핫 인코딩)
            input_cmd_data = np.zeros((1, 4), dtype=np.float32)
            input_cmd_data[0, current_command] = 1.0

            # 모델 추론
            interpreter.set_tensor(img_idx, input_data)
            interpreter.set_tensor(cmd_idx, input_cmd_data)
            interpreter.invoke()
            raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # 결과 필터링
            alpha = 0.8
            smoothed_prediction = (1 - alpha) * smoothed_prediction + alpha * raw_prediction
            
            sensitive = 0.08
            Kp = 2.5
            
            if abs(smoothed_prediction) <= sensitive:
                P = 0.0
            else:
                P = np.clip(smoothed_prediction * Kp, -1.0, 1.0)
            
            # 방향 텍스트 설정 및 모터 제어 변수 계산
            if P == 0.0:
                left_pwm = speedSet
                right_pwm = speedSet
                dir_str = f"⬆️ 직진 (AI: {raw_prediction:+.2f})"
            elif P > 0:
                left_pwm = speedSet
                right_pwm = speedSet * (1.0 - P)
                dir_str = f"➡️ 우회전 (P: {P:+.2f})"
            else:
                left_pwm = speedSet * (1.0 + P)
                right_pwm = speedSet
                dir_str = f"⬅️ 좌회전 (P: {P:+.2f})"

            # 출력: 터미널 한 줄에서 덮어쓰기로 갱신 (캐리지 리턴 \r 활용)
            # 여유 공백을 추가하여 이전의 긴 문자열이 남아있지 않게 덮어씀
            # 입력 방해를 줄이기 위해 상태 변화 또는 0.5초 주기로만 출력
            cmd_str = ["C0:외곽", "C1:좌회전", "C2:우회전", "C3:직진"][current_command]
            current_time = time.time()
            if current_command != last_cmd or latest_sign_text != last_sign or (current_time - last_print_time) > 0.5:
                sys.stdout.write(f"\r[모드] {cmd_str:<10} | [차선] {dir_str:<25} | [표지판] {latest_sign_text:<25}     ")
                sys.stdout.flush()
                last_cmd = current_command
                last_sign = latest_sign_text
                last_print_time = current_time

            set_motors(left_pwm, right_pwm)
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        # 종료 시 스레드 및 카메라 안전하게 정리
        running = False
        print("\n\n주행을 안전하게 종료합니다.")
        stop()
        if 'camera' in locals() and camera.isOpened():
            camera.release()

if __name__ == '__main__':
    main()
