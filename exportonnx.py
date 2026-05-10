from ultralytics import YOLO

# 1. 방금 학습이 끝난 에이스 모델 로드 (경로 주의)
model = YOLO('/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/runs/detect/4WD_Final_Project/train_data_640/weights/best.pt')

# 2. ONNX 형식으로 내보내기 
# (simplify와 opset=13 옵션은 라즈베리파이 같은 기기에서 모델이 더 빠르고 안정적으로 돌아가게 해줍니다)
success = model.export(format='onnx', simplify=True, opset=13)

print("🚀 ONNX 파일 변환 완료!")