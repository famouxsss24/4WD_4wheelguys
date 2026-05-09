from ultralytics import YOLO

# ==========================================
# 📍 1. 끝판왕 모델 불러오기
# ==========================================
# 아까 학습하다가 멈췄을 때 저장된 최고의 성적표(best.pt) 경로입니다.
best_model_path = '/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/runs/detect/4WD_Final_Project/train_1045_data-2/weights/best.pt'
model = YOLO(best_model_path)

# ==========================================
# 📝 2. 실전 테스트(수능) 설정 및 실행
# ==========================================
yaml_path = '/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/Final_Dataset/data.yaml'

print("🚀 드디어 실전 테스트(Test)를 시작합니다! AI가 낯선 사진들을 분석합니다...")

# model.val() 함수에서 split='test'를 주면 최종 평가 모드로 작동합니다.
metrics = model.val(
    data=yaml_path,
    split='test',      # train이나 val이 아닌 'test' 사진만 사용
    device='mps',      # 맥북 M3 Pro 가속
    plots=True         # 결과 그래프, 오차 행렬, 정답 비교 사진 자동 저장
)

# ==========================================
# 📊 3. 최종 결과 출력
# ==========================================
print("\n✅ 모든 테스트가 완료되었습니다!")
print("-" * 40)
print(f"🏆 최종 성적표 (mAP50) : {metrics.box.map50:.4f}")
print(f"🎯 깐깐한 성적 (mAP50-95) : {metrics.box.map:.4f}")
print("-" * 40)
print("📂 '4WD_Final_Project' 폴더 안의 새로 생긴 'val' 폴더에 가보시면")
print("AI가 시험지(사진)에 어떻게 박스를 쳤는지 직접 눈으로 확인할 수 있습니다!")