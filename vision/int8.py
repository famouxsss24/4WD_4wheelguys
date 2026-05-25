from ultralytics import YOLO

# 1. 학습된 원본 PyTorch 모델 로드
model = YOLO("/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/runs/detect/4WD_Final_Project/train_1045_data/weights/best.pt") # 학습 완료된 pt 파일 경로 입력

# 2. INT8 양자화를 적용하여 TFLite로 내보내기
model.export(
    format="tflite",
    int8=True,       # 💡 핵심: 8-bit 정수 양자화 활성화
    imgsz=640,       # 💡 라즈베리파이 최적화를 위해 해상도를 640으로 고정해서 내보냄
    data="/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/Final_Dataset/data.yaml" # 💡 캘리브레이션 데이터 (매우 중요!)
)

print("✅ INT8 TFLite 변환이 완료되었습니다!")