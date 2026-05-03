print("🚀 학습 프로세스 시작 시도 중...")
from ultralytics import YOLO

# 1. 모델 로드 (방금 다운로드된 yolo11n.pt 사용)
model = YOLO('yolo11n.pt')

ame='first_830_train'

# 2. M3 Pro 맞춤형 학습 시작
model.train(
    # 기본 설정
    data='/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/4wheels.yolov11/data.yaml',
    verbose=True,
    epochs=100,
    imgsz=640,
    device='mps',
    cache=True,
    batch=32,
    workers=8,
    project='4WD_AI_Project',
    name='first_830_train',

    # 데이터 증강(Augmentation) 파라미터 추가
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5,
    translate=0.05,
    scale=0.1,
    fliplr=0.0,  # 자율주행 시 좌우 반전이 위험할 수 있으므로 0 유지 권장
    flipud=0.0   # 상하 반전 역시 0 유지 권장
)
