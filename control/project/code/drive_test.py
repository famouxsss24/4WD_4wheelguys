from gpiozero import PWMOutputDevice, DigitalOutputDevice
import time
import sys
import select
import tty
import termios

# ==========================================
# 1. 모터 핀 설정 및 제어 함수
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

# ==========================================
# 터미널 키보드 입력 감지 함수
# ==========================================
def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    print("====================================")
    print("🚗 Starting Driving Test (No Data Collection)!")
    print("w: 직진, s: 후진, x/스페이스: 정지")
    print("q/e: 완만한 좌/우회전 (복원 주행용)")
    print("a/d: 급격한 좌/우회전 (급커브용)")
    print("k: 종료")
    print("====================================")
    
    old_settings = termios.tcgetattr(sys.stdin)
    
    speedSet = 0.5
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while True:
            if isData():
                key = sys.stdin.read(1)
                
                if key == 'k': 
                    break
                
                # ⬆️ 직진
                elif key == 'w':
                    forward(speedSet)
                
                # ==========================================
                # 💡 [핵심 2] 다단 기어 키보드 맵핑 (a, d, q, e)
                # ==========================================
                # ↖️ 완만한 좌회전 (복원 주행용)
                elif key == 'q':
                    set_motors(speedSet*SOFT_CURVE_FACTOR, speedSet, "forward", "forward")
                
                # ↗️ 완만한 우회전 (복원 주행용)
                elif key == 'e':
                    set_motors(speedSet, speedSet*SOFT_CURVE_FACTOR, "forward", "forward")

                # ⬅️ 급격한 좌회전 (급커브용)
                elif key == 'a':
                    set_motors(speedSet*HARD_CURVE_FACTOR, speedSet, "forward", "forward")

                # ➡️ 급격한 우회전 (급커브용)
                elif key == 'd':
                    set_motors(speedSet, speedSet*HARD_CURVE_FACTOR, "forward", "forward")

                # ⬇️ 후진 및 정지
                elif key == 's':
                    stop()
                    time.sleep(0.1)
                    backward(speedSet)
                elif key in ['x', ' ']:
                    stop()

            time.sleep(0.05)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nEnding driving test.")
        stop()

if __name__ == '__main__':
    main()
