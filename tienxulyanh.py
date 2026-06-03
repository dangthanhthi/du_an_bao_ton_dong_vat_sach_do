import os
import shutil
import sys
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import splitfolders
from PIL import Image

# Reconfigure stdout for UTF-8 on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("="*65)
print("BƯỚC 2: TIỀN XỬ LÝ ẢNH, CHIA DỮ LIỆU & CÂN BẰNG TẬP TRAIN (7:2:1)")
print("="*65)

# --- 1. TỰ ĐỘNG DỌN DẸP & KHÔI PHỤC DỮ LIỆU GỐC ---
try:
    from clean_dataset import clean_and_reset_dataset
    clean_and_reset_dataset()
except Exception as e:
    print(f"[*] Cảnh báo khi dọn dẹp dữ liệu gốc: {e}")

# --- 2. ĐƯỜNG DẪN DỮ LIỆU ---
workspace_dir = os.path.dirname(os.path.abspath(__file__))
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# Tìm DATA_DIR trong workspace trước, fallback ra Desktop
DATA_DIR = os.path.join(workspace_dir, "Endangered_Animals_300_Unique")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(desktop_path, "Endangered_Animals_300_Unique")

SPLIT_DIR = os.path.join(desktop_path, "Dataset_ResNet18_Ready")

# --- 3. DỌN DẸP THƯ MỤC CHIA CŨ ---
if os.path.exists(SPLIT_DIR):
    print("[*] Đang làm sạch thư mục chia cũ để chia lại dữ liệu độc lập...")
    shutil.rmtree(SPLIT_DIR)

# --- 4. CHIA TẬP DỮ LIỆU (70% Train - 20% Valid - 10% Test) ---
# Việc chia từ dữ liệu gốc CHƯA augment đảm bảo tập Val và Test hoàn toàn sạch
print(f"[*] Đang tiến hành chia dữ liệu gốc (Train: 70%, Val: 20%, Test: 10%)...")
splitfolders.ratio(DATA_DIR, output=SPLIT_DIR, seed=42, ratio=(0.7, 0.2, 0.1), group_prefix=None)
print(" Đã chia tập dữ liệu gốc thành công!")

# --- 5. CÂN BẰNG TẬP HUẤN LUYỆN (CHỈ AUGMENT TRÊN FOLDER TRAIN) ---
print("\n" + "="*60)
print("[*] TIẾN HÀNH CÂN BẰNG TẬP TRAIN (KHÔNG LÀM RÒ RỈ VÀO VAL/TEST)")
print("="*60)

train_dir = os.path.join(SPLIT_DIR, 'train')
TARGET_TRAIN_COUNT = 210 # 70% của 300 ảnh mục tiêu

augmentation_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),            
    transforms.RandomRotation(degrees=(-15, 15)),      
    transforms.ColorJitter(brightness=0.15, contrast=0.15), 
])

classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]

for cls in classes:
    class_path = os.path.join(train_dir, cls)
    existing_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(existing_images)
    
    if current_count >= TARGET_TRAIN_COUNT:
        print(f"  + {cls}: Đã có {current_count} ảnh train -> Bỏ qua.")
        continue
        
    needed = TARGET_TRAIN_COUNT - current_count
    print(f"  + {cls}: Đang có {current_count} ảnh train -> Sinh thêm {needed} ảnh augment...")
    
    generated_count = 0
    source_images = existing_images.copy()
    
    while generated_count < needed:
        random_img_name = random.choice(source_images)
        img_path = os.path.join(class_path, random_img_name)
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                augmented_img = augmentation_pipeline(img)
                
                new_filename = f"AUG_TRAIN_{generated_count}_{random_img_name}"
                save_path = os.path.join(class_path, new_filename)
                
                augmented_img.save(save_path, format='JPEG', quality=95)
                generated_count += 1
        except Exception as e:
            pass

print(" Cân bằng dữ liệu tập Train hoàn tất!")

# --- 6. ĐỊNH NGHĨA BIẾN ĐỔI (TRANSFORMS) CHUẨN TRONG PYTORCH ---
# Bổ sung RandomRotation và ColorJitter vào online train transforms để tăng khả năng tổng quát hóa
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),                               
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),                          
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),                          
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 7. KHỞI TẠO DATALOADER ---
BATCH_SIZE = 32

image_datasets = {x: datasets.ImageFolder(os.path.join(SPLIT_DIR, x), data_transforms[x]) 
                  for x in ['train', 'val', 'test']}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

print("\nTÓM TẮT DỮ LIỆU SAU KHI XỬ LÝ (ĐÃ FIX DATA LEAKAGE):")
print(f"  + Số lượng lớp (Classes): {len(class_names)}")
print(f"  + Tên các loài: {class_names}")
print(f"  + Tập Huấn luyện (Train) [Đã augment cân bằng]: {dataset_sizes['train']} ảnh ({TARGET_TRAIN_COUNT} ảnh/loài)")
print(f"  + Tập Kiểm tra (Valid) [Chỉ chứa ảnh gốc sạch]: {dataset_sizes['val']} ảnh")
print(f"  + Tập Đánh giá (Test) [Chỉ chứa ảnh gốc sạch]: {dataset_sizes['test']} ảnh")
print("\n HOÀN TẤT TIỀN XỬ LÝ DỮ LIỆU CHUẨN KHOA HỌC!")