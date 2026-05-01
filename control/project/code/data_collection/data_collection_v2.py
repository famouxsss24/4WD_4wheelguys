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
import glob

# ==========================================
# 1. 모터 핀 설정 및 제어 함수 (기존과 동일)
# ==========================================
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

# ==========================================
# 💡 [핵심 1] 다단 기어 커브 팩터 설정
# ==========================================
SOFT_CURVE_FACTOR = 0.6  # 완만한 커브용 (안쪽 바퀴가 60% 속도로 돎)
HARD_CURVE_FACTOR = 0.1  # 급격한 커브용 (안쪽 바퀴가 10% 속도로 돎)

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
# 기존 forward_left, forward_right 삭제 (다단 기어 팩터 사용으로 대체)

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
    print("Starting Lightweight Data Collection v2!")
    print("====================================")
    
    old_settings = termios.tcgetattr(sys.stdin)
    
    # Initialize variables that are used inside the while loop
    frame_count = 0
    speedSet = 0.5
    carState = None
    
    # 기존 파일들을 확인하여 덮어쓰기 방지 (마지막 번호에 이어서 저장)
    existing_files = glob.glob(os.path.join(save_dir, "video_*.jpg"))
    if existing_files:
        try:
            indices = [int(os.path.basename(f).split('_')[1]) for f in existing_files]
            i = max(indices) + 1
        except Exception:
            i = 0
    else:
        i = 0
        
    print(f"📌 저장 인덱스 (덮어쓰기 방지): {i:05d}번부터 시작합니다.")
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while camera.isOpened():
            if isData():
                key = sys.stdin.read(1)
                
                if key == 'k': break
                
                # ⬆️ 직진
                elif key == 'w':
                    carState = (int(speedSet*100), int(speedSet*100))
                    forward(speedSet)
                
                # ==========================================
                # 💡 [핵심 2] 다단 기어 키보드 맵핑 (a, d, q, e)
                # ==========================================
                # ↖️ 완만한 좌회전 (복원 주행용)
                elif key == 'q':
                    carState = (int(speedSet*SOFT_CURVE_FACTOR*100), int(speedSet*100))
                    set_motors(speedSet*SOFT_CURVE_FACTOR, speedSet, "forward", "forward")
                
                # ↗️ 완만한 우회전 (복원 주행용)
                elif key == 'e':
                    carState = (int(speedSet*100), int(speedSet*SOFT_CURVE_FACTOR*100))
                    set_motors(speedSet, speedSet*SOFT_CURVE_FACTOR, "forward", "forward")

                # ⬅️ 급격한 좌회전 (급커브용)
                elif key == 'a':
                    carState = (int(speedSet*HARD_CURVE_FACTOR*100), int(speedSet*100))
                    set_motors(speedSet*HARD_CURVE_FACTOR, speedSet, "forward", "forward")

                # ➡️ 급격한 우회전 (급커브용)
                elif key == 'd':
                    carState = (int(speedSet*100), int(speedSet*HARD_CURVE_FACTOR*100))
                    set_motors(speedSet, speedSet*HARD_CURVE_FACTOR, "forward", "forward")

                # ⬇️ 후진 및 정지 (유지)
                elif key == 's':
                    carState = None 
                    stop()
                    time.sleep(0.1)
                    backward(speedSet)
                elif key in ['x', ' ']:
                    carState = None
                    stop()

            # --- [이미지 전처리 최적화] ---
            _, image = camera.read()
            
            # 이전 대화 기록 확인 결과: mycamera.py가 "RGB888" 설정이어도 
            # 라즈베리파이 하드웨어 특성상 Numpy 배열로 넘어올 때 실제로는 BGR로 들어오는 현상이 있었습니다!
            # 따라서 들어온 BGR을 다시 RGB로 정상화하기 위해 채널을 뒤집어줍니다.
            image = image[:, :, ::-1]
            
            # 1. cv2.flip 대신 Numpy 슬라이싱으로 상하좌우 반전
            image = image[::-1, ::-1, :]
            frame_count += 1
            
            # 초당 수집량 증가: 기존 5프레임당 1장(4FPS) -> 2프레임당 1장(10FPS)으로 변경
            if carState is not None and frame_count % 2 == 0:
                height = image.shape[0]
                
                # ==========================================
                # 💡 [핵심 3] 시야 내리기 (코너 파고들기 방지)
                # ==========================================
                # 기존 int(height/2) 에서 0.6 또는 0.65로 늘려서 차체 바로 앞만 보게 합니다.
                save_image = image[int(height*0.60):, :, :]
                
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
