from ultralytics import YOLO
import os

# 1. 모델 로드 (⭐ 중요: 가장 최근 학습된 97% 짜리 에이스 모델 경로로 변경!)
model = YOLO('runs/detect/4WD_AI_Project/first_830_train-2/weights/best.pt')

# 2. 원본 2,400장 사진 경로
raw_images_path = '/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/raw_data/'

# [진단] 폴더 존재 여부 및 파일 확인
if not os.path.exists(raw_images_path):
    print(f"❌ 에러: {raw_images_path} 폴더가 존재하지 않습니다.")
else:
    files = [f for f in os.listdir(raw_images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🔍 폴더 연결 성공! 사진 {len(files)}장을 찾았습니다.")

# 3. 자동 라벨링 실행 (표지판 특화 옵션 적용)
model.predict(
    source=raw_images_path,
    
    # 🛡️ 표지판 & 방향(상하좌우) 보호를 위한 핵심 옵션 🛡️
    conf=0.7,          # ⭐ 70% 이상 확신할 때만! (어설픈 건 아예 안 잡는 게 나음)
    augment=False,      # ⭐ 예측 시 이미지 뒤집기(TTA) 절대 금지! (좌우 헷갈림 원천 차단)
    imgsz=640,          # ⭐ 학습할 때와 똑같은 해상도(640)로 바라보게 해서 정확도 극대화
    
    # 기본 저장 옵션
    save_txt=True,      # .txt 라벨 파일 생성
    save=True,          # 라벨링된 사진도 저장 (나중에 눈으로 확인하기 좋음)
    save_conf=True,     # 라벨 파일에 확률값 포함 (나중에 분석할 때 유용)
    device='mps',       # M3 Pro GPU 사용
    
    # 📂 새출발하는 기분으로 폴더 이름 변경
    project='4WD_AutoLabel_FreshStart',
    name='batch_2400'
)

print("✅ 방향 감각을 갖춘 스마트 자동 라벨링 완료!")
print(f"👉 확인 경로: 4WD_AutoLabel_FreshStart/batch_2400/")