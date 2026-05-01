from gpiozero import PWMOutputDevice, DigitalOutputDevice
from PIL import Image  # cv2.imwrite 대신 사용할 가벼운 이미지 라이브러리
import time
import os
import threading
import queue
import sys
import select
import tty
import termios
import mycamera

# ==========================================
# 1. 모터 핀 설정 및 제어 함수 (기존과 동일)
# ==========================================
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

CURVE_SPEED_FACTOR = 0.3

def set_motors(left_speed, right_speed, left_dir, right_dir):
    if left_dir == "forward":
        AIN1.value, AIN2.value = 0, 1
    else:
        AIN1.value, AIN2.value = 1, 0
        
    if right_dir == "forward":
        BIN1.value, BIN2.value = 0, 1
    else:
        BIN1.value, BIN2.value = 1, 0
        
    PWMA.value = left_speed
    PWMB.value = right_speed

def stop():
    PWMA.value, PWMB.value = 0, 0

def forward(speed=0.5): set_motors(speed, speed, "forward", "forward")
def backward(speed=0.5): set_motors(speed, speed, "backward", "backward")
def turn_left(speed=0.5): set_motors(speed, speed, "backward", "forward")
def turn_right(speed=0.5): set_motors(speed, speed, "forward", "backward")
def forward_left(speed=0.5): set_motors(speed*CURVE_SPEED_FACTOR, speed, "forward", "forward")
def forward_right(speed=0.5): set_motors(speed, speed*CURVE_SPEED_FACTOR, "forward", "forward")

# ==========================================
# 2. 비동기 이미지 저장 스레드 (PIL 사용)
# ==========================================
save_queue = queue.Queue()

def image_saver_thread():
    """Numpy 배열을 받아 PIL을 이용해 매우 가볍게 저장합니다."""
    while True:
        data = save_queue.get()
        if data is None:
            break
        filename, image_array = data

        # Picamera2는 RGB888 포맷으로 이미 RGB 순서로 데이터를 줌
        # 채널 변환 없이 그대로 저장
        Image.fromarray(image_array).save(filename)
        save_queue.task_done()

# ==========================================
# 3. 터미널 키보드 입력 감지 함수 (기존과 동일)
# ==========================================
def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    camera = mycamera.MyPiCamera(320, 240)
    save_dir = "/home/fourwheel/Desktop/project/code/video"

    #if not os.path.exists(save_dir):
        #os.makedirs(save_dir)
        #print(f"New folder created: {save_dir}")
    saver = threading.Thread(target=image_saver_thread)
    saver.daemon = True
    saver.start()


    print("====================================")
    print("Starting Lightweight Data Collection!")
    print("====================================")
    
    old_settings = termios.tcgetattr(sys.stdin)
    
    # Initialize variables that are used inside the while loop
    frame_count = 0
    speedSet = 0.5
    carState = None
    i = 0
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while camera.isOpened():
            if isData():
                key = sys.stdin.read(1)
                
                # ... [키보드 제어 로직은 기존 코드와 완벽히 동일하므로 생략 없이 그대로 유지] ...
                if key == 'k': break
                # elif key == '2':
                #     speedSet = min(1.0, speedSet + 0.1)
                #     print(f"Max Speed: {int(speedSet*100)}% \r", end="")
                # elif key == '1':
                #     speedSet = max(0.2, speedSet - 0.1)
                #     print(f"Max Speed: {int(speedSet*100)}% \r", end="")
                elif key == 'w':
                    carState = (int(speedSet*100), int(speedSet*100))
                    forward(speedSet)
                elif key == 'a':
                    stop()
                    time.sleep(0.1) 
                    carState = (int(-speedSet*100), int(speedSet*100))
                    turn_left(speedSet)
                elif key == 'd':
                    stop()
                    time.sleep(0.1)
                    carState = (int(speedSet*100), int(-speedSet*100))
                    turn_right(speedSet)
                elif key == 'q':
                    carState = (int(speedSet*CURVE_SPEED_FACTOR*100), int(speedSet*100))
                    forward_left(speedSet)
                elif key == 'e':
                    carState = (int(speedSet*100), int(speedSet*CURVE_SPEED_FACTOR*100))
                    forward_right(speedSet)
                elif key == 's':
                    carState = None 
                    stop()
                    time.sleep(0.1)
                    backward(speedSet)
                elif key in ['x', ' ']:
                    carState = None
                    stop()

            # --- [핵심 변경 포인트: 이미지 전처리 최적화] ---
            _, image = camera.read()
            
            # BGR로 들어오는 데이터를 RGB로 변환 (auto_drive.py와 동일하게)
            image = image[:, :, ::-1]
            
            # 1. cv2.flip 대신 Numpy 슬라이싱으로 상하좌우 반전
            image = image[::-1, ::-1, :]
            frame_count += 1
            
            # 초당 수집량 증가: 기존 5프레임당 1장(4FPS) -> 2프레임당 1장(10FPS)으로 변경
            if carState is not None and frame_count % 2 == 0:
                height = image.shape[0]
                
                # 2. 불필요한 연산 제거: 하단 절반 영역(ROI)만 잘라냅니다.
                # 블러, 리사이즈, YUV 변환은 나중에 PC에서 학습 전에 일괄 처리하세요!
                save_image = image[int(height/2):, :, :]
                
                left_pwm, right_pwm = carState
                filename = os.path.join(save_dir, f"video_{i:05d}_L{left_pwm:+04d}_R{right_pwm:+04d}.jpg") # png보단 jpg가 저장 속도가 빠름
                
                save_queue.put((filename, save_image))
                i += 1

            time.sleep(0.05)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nEnding driving and data collection.")
        stop()
        
        save_queue.put(None)
        saver.join()
        
        if 'camera' in locals() and camera.isOpened():
            camera.release()

if __name__ == '__main__':
    main()