import numpy as np
import ai_edge_litert.interpreter as tflite
import time
import mycamera
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# ==========================================
# 1. 모터 핀 설정 및 제어 함수 (수집 코드와 100% 동일)
# ==========================================
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

CURVE_SPEED_FACTOR = 0.3

def set_motors(left_speed, right_speed, left_dir, right_dir):
    if left_dir == "forward": AIN1.value, AIN2.value = 0, 1
    else: AIN1.value, AIN2.value = 1, 0
        
    if right_dir == "forward": BIN1.value, BIN2.value = 0, 1
    else: BIN1.value, BIN2.value = 1, 0
        
    PWMA.value = left_speed
    PWMB.value = right_speed

def stop():
    PWMA.value, PWMB.value = 0, 0

def forward(speed=0.5): set_motors(speed, speed, "forward", "forward")
def forward_left(speed=0.5, curve_factor=1.0): 
    # curve_factor(AI 예측값)가 클수록 안쪽 바퀴 속도를 줄여 더 급하게 꺾습니다.
    set_motors(speed * max(1.0 - curve_factor, 0.0), speed, "forward", "forward")

def forward_right(speed=0.5, curve_factor=1.0): 
    set_motors(speed, speed * max(1.0 - curve_factor, 0.0), "forward", "forward")

# ==========================================
# 2. AI 모델(뇌) 불러오기
# ==========================================
# TFLite 파일 경로를 정확히 맞춰주세요. (같은 폴더에 있다고 가정)
model_path = "code/my_rc_car_model_ver1.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def main():
    camera = mycamera.MyPiCamera(320, 240)
    speedSet = 0.5  # 수집할 때 썼던 그 속도 그대로!

    print("====================================")
    print("🚀 완벽한 AI 자율주행을 시작합니다! (Ctrl+C로 종료)")
    print("====================================")

    try:
        while camera.isOpened():
            # 1. 사진 찰칵
            _, image = camera.read()
            
            # 2. 전처리 (수집 코드와 토씨 하나 안 틀리고 똑같이!)
            image = image[:, :, ::-1]       # BGR -> RGB 변환
            image = image[::-1, ::-1, :]    # 상하좌우 반전
            height = image.shape[0]
            roi_image = image[int(height/2):, :, :] # 하단 절반(ROI) 자르기
            
            # 3. 모델에 넣기 위한 정규화 (0~255 -> 0.0~1.0) 및 차원 추가
            # AI는 1장씩 먹더라도 묶음(Batch) 형태인 [1, 120, 320, 3]을 원합니다.
            input_data = np.float32(roi_image) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            # 4. AI에게 조향값 물어보기 (추론)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # 💡 [영점 조절] 이번 모델은 너무 완벽해서 일단 순정으로 갑니다.
            # 만약 살짝 쏠린다면 prediction = prediction - 0.1 처럼 빼주시면 됩니다.
            
            # 5. AI의 판단에 따라 모터 움직이기
            sensitive = 0.2
            prediction = np.clip(prediction, -1.0, 1.0)
            abs_prediction = abs(prediction)
            Kp = 0.8
            if abs_prediction <= sensitive:
                print(f"⬆️ 직진! (예측값: {prediction:.2f})")
                forward(speedSet)
            else:
                # Kp를 곱해서 모터에 들어갈 최종 커브 팩터를 계산합니다.
                final_curve_factor = abs_prediction * Kp
                
                if prediction < 0:
                    print(f"⬅️ 좌회전! (예측값: {prediction:.2f} / 감쇠적용: {final_curve_factor:.2f})")
                    forward_left(speedSet, final_curve_factor)
                else:
                    print(f"➡️ 우회전! (예측값: {prediction:.2f} / 감쇠적용: {final_curve_factor:.2f})")
                    forward_right(speedSet, final_curve_factor)    
            # 모터 과부하 및 너무 잦은 명령 방지
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n주행을 안전하게 종료합니다.")

    finally:
        stop()
        if 'camera' in locals() and camera.isOpened():
            camera.release()

if __name__ == '__main__':
    main()