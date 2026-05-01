from gpiozero import PWMOutputDevice, DigitalOutputDevice
import cv2
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

# Speed reduction factor for curve driving (default: 0.3)
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
# 2. 비동기 이미지 저장 스레드
# ==========================================
save_queue = queue.Queue()

def image_saver_thread():
    """메인 주행 루프에 영향을 주지 않고 배경에서 이미지를 저장합니다."""
    while True:
        data = save_queue.get()
        if data is None: 
            break # None이 들어오면 스레드 종료
        filename, image = data
        cv2.imwrite(filename, image)
        save_queue.task_done()

# ==========================================
# 3. 터미널 키보드 입력 감지 함수 (cv2.waitKey 대체)
# ==========================================
def isData():
    """터미널에 입력된 키가 있는지 확인합니다 (논블로킹)."""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    camera = mycamera.MyPiCamera(320, 240)
    save_dir = "/home/fourwheel/Desktop/test/code/video"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"New folder created: {save_dir}")

    # 이미지 저장 스레드 시작
    saver = threading.Thread(target=image_saver_thread)
    saver.daemon = True
    saver.start()

    i = 0
    frame_count = 0
    carState = None
    speedSet = 0.3

    print("====================================")
    print("Starting Lightweight Data Collection!")
    print("====================================")
    print("w: Forward, s: Backward, a: Turn Left, d: Turn Right")
    print("q: Curve Left, e: Curve Right, x/space: Stop")
    print("Press 'k' to exit. (No video output)")

    # 터미널 세팅 변경 (엔터키 없이 바로 입력받도록 설정)
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while camera.isOpened():
            # 키보드 입력 처리 (논블로킹)
            if isData():
                key = sys.stdin.read(1)
                
                if key == 'k':
                    break
                elif key == '2':
                    speedSet = min(1.0, speedSet + 0.1)
                    print(f"Max Speed: {int(speedSet*100)}% \r", end="")
                elif key == '1':
                    speedSet = max(0.2, speedSet - 0.1)
                    print(f"Max Speed: {int(speedSet*100)}% \r", end="")
                elif key == 'w':
                    carState = (int(speedSet*100), int(speedSet*100))
                    forward(speedSet)
                    
                #역전압 방지: 회전 시 잠깐 멈춤
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
                    
                #역전압 방지: 후진 시 잠깐 멈춤
                elif key == 's':
                    carState = None 
                    stop()
                    time.sleep(0.1)
                    backward(speedSet)
                    
                elif key in ['x', ' ']:
                    carState = None
                    stop()

            # 카메라 읽기
            _, image = camera.read()
            image = cv2.flip(image, -1)
            frame_count += 1
            
            #저장할 타이밍에만 영상 전처리 수행
            if carState is not None and frame_count % 5 == 0:
                height, _, _ = image.shape
                save_image = image[int(height/2):,:,:]
                save_image = cv2.cvtColor(save_image, cv2.COLOR_BGR2YUV)
                save_image = cv2.GaussianBlur(save_image, (3,3), 0)
                save_image = cv2.resize(save_image, (200,66))
                
                left_pwm, right_pwm = carState
                filename = os.path.join(save_dir, f"video_{i:05d}_L{left_pwm:+04d}_R{right_pwm:+04d}.png")
                
                #큐에 던지기
                save_queue.put((filename, save_image))
                i += 1

            #CPU 폭주 방지용 브레이크 (약 30FPS 제한)
            time.sleep(0.05)

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nEnding driving and data collection.")
        stop()
        
        # 남은 저장 작업 완료 대기
        save_queue.put(None)
        saver.join()
        
        if 'camera' in locals() and camera.isOpened():
            camera.release()

if __name__ == '__main__':
    main()


"""
1.main loop안에서 파일 저장을 하기 때문에 만약 조종 지연이 심해지면 Thread 분리해서 따로 빼놓을 것
2.데이터를 수집 후 직진 데이터와 커브 데이터의 수를 조율할 것, 직진 데이터가 너무 많으면 안됨 
3.가능하면 차선에서 벗어났을 때 어떻게 해야하는지에 대한 데이터도 수집해볼 것
https://play2-gound.tistory.com/52
"""