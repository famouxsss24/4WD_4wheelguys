from ultralytics import YOLO

# 1. 모델 로드 (YOLOv11 나노 모델)
model = YOLO('yolo11n.pt')

# 2. 학습 시작
model.train(
    # 📍 경로 설정 (수정 완료된 절대 경로)
    data='/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/Final_Dataset/data.yaml',
    
    # ⚙️ 기본 학습 설정
    epochs=150,         # (수정) 혹독한 증강을 소화하기 위해 에포크를 늘립니다.
    patience=20,        # (추가) 20번 연속으로 성적이 안 오르면 알아서 조기 종료합니다.
    imgsz=[480, 640],          # [480, 640]으로 쓰셔도 되지만, 보통 정사각형(640)이 표준입니다.
    device='mps',       # 맥북 M3 Pro GPU 가속
    batch=32,           # 한 번에 처리할 사진 수 (M3 Pro 메모리에 맞게 16~32 추천)
    workers=8,          # 데이터 로딩 속도 향상
    cache=True,         # RAM에 데이터를 올려둬서 학습 속도 부스팅
    verbose=True,
    
    # 🛡️ 데이터 증강 설정 (방향 표지판 보호 및 실전 훈련 강화)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5,
    translate=0.05,
    scale=0.5,        # (수정) 원거리/근거리 표지판 크기 변화 대응
    fliplr=0.0,       # (유지) 방향 표지판 보호 (절대 필수)
    flipud=0.0,       # (유지) 상하 반전 금지
    erasing=0.4,      # (추가) 표지판 일부가 가려져도 인식하도록 훈련
    perspective=0.0005, # (추가) 다가갈 때 비스듬하게 보이는 각도 학습
    mosaic=1.0,       # (추가) 작은 표지판 인식률 극대화
    
    # 📂 저장 위치 설정
    project='4WD_Final_Project',
    name='train_1045_data'
)

print("✅ 학습이 성공적으로 시작되었습니다!")