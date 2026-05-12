import numpy as np
import cv2
import ai_edge_litert.interpreter as tflite
import time
import sys
import select
import tty
import termios
import threading
import mycamera
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# ==========================================
# 1. 모터 핀 설정 (기존과 동일)
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
# 💡 [핵심 1] 실시간 CIL 커맨드 입력 스레드
# ==========================================
# 자율주행 중에도 키보드(1,2,3,4)를 눌러 목적지를 동적으로 바꿀 수 있게 합니다.
current_command = 0  # 초기값: C0 (외곽 주행)
running = True

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def command_listener():
    global current_command, running
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while running:
            if isData():
                key = sys.stdin.read(1)
                if key == '1':
                    current_command = 0
                    print("\n🎯 [지시 수신] C0: 외곽 주행 모드")
                elif key == '2':
                    current_command = 1
                    print("\n🎯 [지시 수신] C1: 다음 교차로에서 좌회전")
                elif key == '3':
                    current_command = 2
                    print("\n🎯 [지시 수신] C2: 다음 교차로에서 우회전")
                elif key == '4':
                    current_command = 3
                    print("\n🎯 [지시 수신] C3: 교차로 직진 통과")
                elif key == 'k':
                    running = False
                    break
            time.sleep(0.1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def main():
    global running, current_command
    camera = mycamera.MyPiCamera(320, 240)
    speedSet = 0.5
    smoothed_prediction = 0.0  

    # ==========================================
    # 2.TFLite 모델 로드
    # ==========================================
    # 저장하신 양자화 모델 경로를 지정하세요.
    model_path = "/home/fourwheel/Desktop/project/model/my_rc_car_cil_model_normal.tflite"
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 💡 [핵심 2] 텐서 인덱스 동적 할당 (매우 중요)
    # TFLite 변환 시 이미지와 커맨드 입력의 순서(0번, 1번)가 랜덤하게 바뀔 수 있습니다.
    # 차원(Shape)의 길이로 이미지(4차원)와 커맨드(2차원)를 안전하게 매핑합니다.
    img_idx = None
    cmd_idx = None
    for detail in input_details:
        if len(detail['shape']) == 4:
            img_idx = detail['index']
        elif len(detail['shape']) == 2:
            cmd_idx = detail['index']

    print("====================================")
    print("🚀 다중 입력(CIL) 자율주행을 시작합니다!")
    print(" [ 1:외곽 | 2:좌회전 | 3:우회전 | 4:직진 | K:종료 ] ")
    print("====================================")

    # 커맨드 리스너 스레드 시작
    cmd_thread = threading.Thread(target=command_listener)
    cmd_thread.daemon = True
    cmd_thread.start()

    try:
        while camera.isOpened() and running:
            _, image = camera.read()
            
            # --- 1. 이미지 전처리 (학습 코드와 100% 동일하게) ---
            image = image[:, :, ::-1] # BGR to RGB
            image = image[::-1, ::-1, :] # 상하좌우 반전
            
            height = image.shape[0]
            roi_image = image[int(height * 0.60):, :, :]
            roi_yuv = cv2.cvtColor(roi_image, cv2.COLOR_RGB2YUV)
            
            # 모델에 넣을 이미지 데이터 준비
            input_img_data = np.float32(roi_yuv) / 255.0
            input_img_data = np.expand_dims(input_img_data, axis=0)

            # --- 2. 커맨드 전처리 (원핫 인코딩) ---
            input_cmd_data = np.zeros((1, 4), dtype=np.float32)
            input_cmd_data[0, current_command] = 1.0

            # --- 3. 모델 추론 (다중 입력 주입) ---
            interpreter.set_tensor(img_idx, input_img_data)
            interpreter.set_tensor(cmd_idx, input_cmd_data)
            interpreter.invoke()
            
            # --- 4. 결과 출력 및 모터 제어 ---
            raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # EMA 필터 (급격한 조향 튐 방지)
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
                # 콘솔 출력이 너무 빠르면 주행에 방해되므로 출력은 최소화하거나 지우셔도 좋습니다.
            elif P > 0:
                left_pwm, right_pwm = speedSet, speedSet * (1.0 - P)
            else:
                left_pwm, right_pwm = speedSet * (1.0 + P), speedSet

            set_motors(left_pwm, right_pwm)
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print("\n🏁 주행을 안전하게 종료합니다.")
        stop()
        if 'camera' in locals() and camera.isOpened():
            camera.release()

if __name__ == '__main__':
    main()