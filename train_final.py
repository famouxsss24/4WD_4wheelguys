from ultralytics import YOLO

# 1. 모델 로드 (YOLOv11 나노 모델)
model = YOLO('yolo11n.pt')

# 2. 학습 시작
model.train(
    # 📍 경로 설정 (수정 완료된 절대 경로)
    data='/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/4wheels.yolov11/data.yaml',
    
    # ⚙️ 기본 학습 설정
    epochs=50,
    imgsz=[480, 640],          # [480, 640]으로 쓰셔도 되지만, 보통 정사각형(640)이 표준입니다.
    device='mps',       # 맥북 M3 Pro GPU 가속
    batch=32,           # 한 번에 처리할 사진 수 (M3 Pro 메모리에 맞게 16~32 추천)
    workers=8,          # 데이터 로딩 속도 향상
    cache=True,         # RAM에 데이터를 올려둬서 학습 속도 부스팅
    verbose=True,   
    
    # 🛡️ 데이터 증강 설정 (방향 표지판 보호)
    fliplr=0.0,         # ⭐ 핵심: 좌우 반전 금지 (left/right 표지판 헷갈림 방지)
    flipud=0.0,         # 상하 반전 금지
    
    # 📂 저장 위치 설정
    project='4WD_Final_Project',
    name='train_1045_data'
)

print("✅ 학습이 성공적으로 시작되었습니다!")