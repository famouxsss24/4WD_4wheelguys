import os
import shutil
from ultralytics import YOLO

# 1. 새로 학습 완료될 원본 PyTorch 모델 로드
best_model_path = "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/runs/detect/vision/runs/detect/4WD_Final_Project/train_data_0526/weights/best.pt"
model = YOLO(best_model_path)

# 저장될 디렉토리 경로 (weights 폴더)
save_dir = os.path.dirname(best_model_path)

print("🚀 1단계: 320 해상도 INT8 양자화 변환 시작...")

# 2. INT8 양자화를 적용하여 TFLite로 내보내기
exported_path = model.export(
    format="tflite",
    int8=True,       # 💡 핵심: 8-bit 정수 양자화 활성화 (라즈베리파이 가속용)
    imgsz=320,       # 💡 해상도를 320으로 고정해서 내보냄
    data="/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/vision/Final_Datasetv4/data.yaml" # 💡 캘리브레이션 데이터
)

# 3. 변환 완료 후 알기 쉬운 이름으로 복사 및 이름 변경
# (예: weights 폴더 바로 아래에 best_320_int8_0526.tflite 로 복사)
target_tflite_path = os.path.join(save_dir, "best_320_int8_0526.tflite")
shutil.copy(exported_path, target_tflite_path)

print(f"✅ INT8 TFLite 변환 및 저장 완료!")
print(f"📂 저장 위치: {target_tflite_path}")