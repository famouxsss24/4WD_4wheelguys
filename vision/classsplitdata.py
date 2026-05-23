import os
import shutil
import random
from collections import defaultdict

# ==========================================
# 📍 1. 경로 설정
# ==========================================
source_folders = [
    {
        "name": "📱 Phone",
        "images": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/phonedata830/train/images",
        "labels": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/phonedata830/train/labels"
    },
    {
        "name": "🍓 Raspberry Pi",
        "images": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/raspberry.yolov11/train/images",
        "labels": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/raspberry.yolov11/train/labels"
    },
    {
        "name": "add_finaldata",
        "images": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/finaldata/train/images",
        "labels": "/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/finaldata/train/labels"
    }
]

output_dir = '/Users/gwonchangbin/univ/2-1/임베디드인공지능최적화/4WD_4wheelguys/Final_Datasetv4'

# 비율 설정
train_ratio = 0.8
valid_ratio = 0.1
valid_extensions = ('.jpg', '.jpeg', '.png')

# 폴더 생성
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

def get_major_class(lbl_path):
    """라벨 파일에서 가장 많이 등장하거나 첫 번째로 등장하는 클래스 ID 반환"""
    if not os.path.exists(lbl_path):
        return -1  # 라벨 없음 (배경)
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return -1
        # 첫 번째 객체의 클래스 ID를 기준으로 삼음 (가장 일반적인 방식)
        return lines[0].split()[0]

def distribute_files(image_list, img_dir, lbl_dir, split_name):
    for img_name in image_list:
        img_src = os.path.join(img_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_src = os.path.join(lbl_dir, txt_name)
        
        img_dst = os.path.join(output_dir, 'images', split_name, img_name)
        txt_dst = os.path.join(output_dir, 'labels', split_name, txt_name)
        
        # 이름 충돌 방지
        if os.path.exists(img_dst):
            unique_id = random.randint(1000, 9999)
            img_dst = os.path.join(output_dir, 'images', split_name, f"dup_{unique_id}_{img_name}")
            txt_dst = os.path.join(output_dir, 'labels', split_name, f"dup_{unique_id}_{txt_name}")
            
        shutil.copy2(img_src, img_dst)
        if os.path.exists(txt_src):
            shutil.copy2(txt_src, txt_dst)
        else:
            open(txt_dst, 'w').close()

# ==========================================
# 🚀 2. 실행 로직
# ==========================================

for folder in source_folders:
    img_dir = folder["images"]
    lbl_dir = folder["labels"]
    
    if not os.path.exists(img_dir):
        print(f"❌ 폴더 없음: {folder['name']}")
        continue

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    random.seed(42)
    random.shuffle(images)

    # --- 클래스별로 이미지 그룹화 ---
    class_map = defaultdict(list)
    
    if folder["name"] == "add_finaldata":
        # finaldata는 클래스가 하나이므로 전체를 하나의 그룹으로 처리
        class_map["single_class"] = images
    else:
        # Phone, Raspberry Pi는 라벨 내용을 확인하여 클래스별로 분류
        for img in images:
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img)[0] + '.txt')
            cls_id = get_major_class(lbl_path)
            class_map[cls_id].append(img)

    # --- 클래스별 8:1:1 분할 및 복사 ---
    print(f"\n📊 {folder['name']} 처리 중...")
    
    for cls_id, cls_images in class_map.items():
        total = len(cls_images)
        if total == 0: continue
        
        idx1 = int(total * train_ratio)
        idx2 = idx1 + int(total * valid_ratio)
        
        train_p = cls_images[:idx1]
        val_p = cls_images[idx1:idx2]
        test_p = cls_images[idx2:]
        
        distribute_files(train_p, img_dir, lbl_dir, 'train')
        distribute_files(val_p, img_dir, lbl_dir, 'val')
        distribute_files(test_p, img_dir, lbl_dir, 'test')
        
        print(f"  - Class [{cls_id}]: Train {len(train_p)}, Val {len(val_p)}, Test {len(test_p)}")

print("\n✅ 클래스 기반 데이터 분할 완료!")