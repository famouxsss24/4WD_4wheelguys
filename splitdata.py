import os
import shutil
import random

# ==========================================
# 📍 1. 원본 폴더들 설정 (경로를 수정해 주세요!)
# ==========================================
source_folders = [
    {
        "name": "📱 Phone",
        "images": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/phone.yolov11/train/images",
        "labels": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/phone.yolov11/train/labels"
    },
    {
        "name": "🍓 Raspberry Pi",
        "images": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/raspberry.yolov11/train/images",
        "labels": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/raspberry.yolov11/train/labels"
    }
]

output_dir = '/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/Final_Dataset'

# ==========================================
# ⚙️ 2. 비율 설정 (Train 80%, Valid 10%, Test 10%)
# ==========================================
train_ratio = 0.8
valid_ratio = 0.1

# 3. 새로운 YOLO 구조 폴더 만들기
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

valid_extensions = ('.jpg', '.jpeg', '.png')

# 4. 파일 복사 함수 (이름 충돌 방지 포함)
def distribute_files(image_list, img_dir, lbl_dir, split_name):
    success_count = 0
    for img_name in image_list:
        img_src = os.path.join(img_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_src = os.path.join(lbl_dir, txt_name)
        
        img_dst = os.path.join(output_dir, 'images', split_name, img_name)
        txt_dst = os.path.join(output_dir, 'labels', split_name, txt_name)
        
        # 이름 충돌 방지 로직
        if os.path.exists(img_dst):
            new_img_name = f"dup_{random.randint(1000, 9999)}_{img_name}"
            new_txt_name = os.path.splitext(new_img_name)[0] + '.txt'
            img_dst = os.path.join(output_dir, 'images', split_name, new_img_name)
            txt_dst = os.path.join(output_dir, 'labels', split_name, new_txt_name)
            
        shutil.copy2(img_src, img_dst)
        
        if os.path.exists(txt_src):
            shutil.copy2(txt_src, txt_dst)
        else:
            open(txt_dst, 'w').close()
            
        success_count += 1
    return success_count

# ==========================================
# 🚀 5. 계층적 분할 (Stratified Split) 핵심 로직
# ==========================================
print("🔍 Stratified Split (기기별 계층화 분할) 시작...\n")

for folder in source_folders:
    img_dir = folder["images"]
    lbl_dir = folder["labels"]
    
    if not os.path.exists(img_dir):
        print(f"❌ 경고: {folder['name']}의 폴더가 없습니다! ({img_dir})")
        continue

    # 해당 기기의 사진만 먼저 싹 가져와서 섞음
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    random.seed(42)
    random.shuffle(images)
    
    # 해당 기기 안에서 8:1:1로 나눔 (이게 Stratified의 핵심!)
    total_count = len(images)
    train_idx = int(total_count * train_ratio)
    valid_idx = train_idx + int(total_count * valid_ratio)
    
    train_imgs = images[:train_idx]
    val_imgs = images[train_idx:valid_idx]
    test_imgs = images[valid_idx:]
    
    print(f"📊 {folder['name']} 분할 (총 {total_count}장) -> Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
    
    # 나눈 비율대로 합산 폴더에 투척
    t_cnt = distribute_files(train_imgs, img_dir, lbl_dir, 'train')
    v_cnt = distribute_files(val_imgs, img_dir, lbl_dir, 'val')
    test_cnt = distribute_files(test_imgs, img_dir, lbl_dir, 'test')

print("\n✅ 모든 기기의 데이터가 비율을 유지한 채 완벽하게 섞였습니다! 'Final_Dataset'을 확인하세요.")