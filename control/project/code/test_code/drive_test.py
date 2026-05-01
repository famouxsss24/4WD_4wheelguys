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
# 2. 터미널 키보드 입력 감지 함수
# ==========================================
def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def main():
    print("====================================")
    print("Starting Drive Test Mode (No Camera)")
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
                # 속도 조절 기능 (필요시 주석 해제)
                # elif key == '2':
                #     speedSet = min(1.0, speedSet + 0.1)
                #     print(f"Max Speed: {int(speedSet*100)}% \r", end="")
                # elif key == '1':
                #     speedSet = max(0.2, speedSet - 0.1)
                #     print(f"Max Speed: {int(speedSet*100)}% \r", end="")
                elif key == 'w':
                    forward(speedSet)
                elif key == 'a':
                    stop()
                    time.sleep(0.1) 
                    turn_left(speedSet)
                elif key == 'd':
                    stop()
                    time.sleep(0.1)
                    turn_right(speedSet)
                elif key == 'q':
                    forward_left(speedSet)
                elif key == 'e':
                    forward_right(speedSet)
                elif key == 's':
                    stop()
                    time.sleep(0.1)
                    backward(speedSet)
                elif key in ['x', ' ']:
                    stop()

            time.sleep(0.05)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nEnding drive test.")
        stop()

if __name__ == '__main__':
    main()
