import numpy as np
import cv2
import ai_edge_litert.interpreter as tflite
import time
import mycamera
from gpiozero import PWMOutputDevice, DigitalOutputDevice

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

model_path = "code/model/my_rc_car_model_ver3.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def main():
    camera = mycamera.MyPiCamera(320, 240)
    speedSet = 0.5
    
    smoothed_prediction = 0.0  

    print("====================================")
    print("🚀 YUV 기반 AI 자율주행을 시작합니다! (Ctrl+C 종료)")
    print("====================================")

    try:
        while camera.isOpened():
            _, image = camera.read()
            
            image = image[:, :, ::-1]
            image = image[::-1, ::-1, :]
            
            height = image.shape[0]
            roi_image = image[int(height * 0.60):, :, :]
            
            roi_yuv = cv2.cvtColor(roi_image, cv2.COLOR_RGB2YUV)
            
            input_data = np.float32(roi_yuv) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            raw_prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            alpha = 0.8
            smoothed_prediction = (1 - alpha) * smoothed_prediction + alpha * raw_prediction
            
            sensitive = 0.08
            Kp = 2.5
            
            if abs(smoothed_prediction) <= sensitive:
                P = 0.0
            else:
                P = np.clip(smoothed_prediction * Kp, -1.0, 1.0)
            
            if P == 0.0:
                left_pwm = speedSet
                right_pwm = speedSet
                print(f"⬆️ 직진 (AI: {raw_prediction:+.2f} -> 필터: {smoothed_prediction:+.2f})")
            elif P > 0:
                left_pwm = speedSet
                right_pwm = speedSet * (1.0 - P)
                print(f"➡️ 우회전 (P: {P:+.2f} | L: {left_pwm:.2f}, R: {right_pwm:.2f})")
            else:
                left_pwm = speedSet * (1.0 + P)
                right_pwm = speedSet
                print(f"⬅️ 좌회전 (P: {P:+.2f} | L: {left_pwm:.2f}, R: {right_pwm:.2f})")

            set_motors(left_pwm, right_pwm)
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n주행을 안전하게 종료합니다.")

    finally:
        stop()
        if 'camera' in locals() and camera.isOpened():
            camera.release()

if __name__ == '__main__':
    main()