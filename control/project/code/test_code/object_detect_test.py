import cv2
from ultralytics import YOLO
import mycamera

# ==========================================
# 1. 팀원이 준 ONNX 모델 로드
# ==========================================
# 팀원분이 주신 onnx 파일 이름을 정확히 적어주세요.
# (처음 실행할 때 내부적으로 최적화하느라 몇 초 정도 걸릴 수 있습니다)
model_path = "code/model/best ver 3.onnx"
model = YOLO(model_path)

# ==========================================
# 2. 카메라 셋팅
# ==========================================
camera = mycamera.MyPiCamera(320, 240) 

print("====================================")
print("🚦 표지판 인식 테스트 시작! (Ctrl+C로 종료)")
print("====================================")

try:
    while camera.isOpened():
        # 1. 사진 찰칵
        _, frame = camera.read()
        
        # ⚠️ 1단계: 카메라가 거꾸로 달려있으므로 상하좌우 반전 (물리적 방향 맞추기)
        frame = frame[::-1, ::-1, :]
        
        # ⚠️ 2단계: 색상 채널 맞추기 (가장 중요!)
        # mycamera는 정상적인 RGB를 줍니다. 하지만 YOLO(Ultralytics) 모듈은
        # 파이썬 Numpy 배열을 입력받을 때 무조건 "이건 OpenCV처럼 BGR이겠지?"라고 가정하고
        # 내부에서 스스로 RGB로 한 번 더 뒤집어버립니다.
        # 따라서, 정상적인 RGB 사진을 그대로 주면 YOLO 내부에서 R과 B가 뒤바뀌어
        # 빨간색 표지판을 파란색으로 착각하게 됩니다!
        # 해결책: YOLO가 내부에서 뒤집을 것을 대비해, 우리가 먼저 BGR로 변환해서 던져줍니다.
        frame_for_yolo = frame[:, :, ::-1]  

        # 2. YOLO 모델에 사진 넣고 추론!
        # verbose=False를 해야 터미널에 쓸데없는 로그가 도배되는 것을 막습니다.
        results = model(frame_for_yolo, verbose=False)

        # 3. 결과 해석 (화면에 표지판이 보였는지 확인)
        for box in results[0].boxes:
            class_id = int(box.cls[0])      # 어떤 표지판인지 (0번, 1번...)
            confidence = float(box.conf[0]) # AI의 확신도 (0.0 ~ 1.0)

            # 💡 신뢰도(확률)가 60% 이상일 때만 출력하도록 필터링 (오작동 방지)
            if confidence > 0.7:
                # 팀원이 Roboflow에서 설정한 클래스 이름(예: 'stop', 'go')을 자동으로 가져옵니다.
                class_name = model.names[class_id]
                
                print(f"인식됨! 👉 [ {class_name} ] (정확도: {confidence*100:.1f}%)")

except KeyboardInterrupt:
    print("\n테스트를 안전하게 종료합니다.")

finally:
    if 'camera' in locals() and camera.isOpened():
        camera.release()