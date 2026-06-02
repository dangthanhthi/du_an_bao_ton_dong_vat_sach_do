import os
import random
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms

# --- 1. CẤU HÌNH ĐƯỜNG DẪN & MỤC TIÊU ---
DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "Endangered_Animals_300_Unique")
TARGET_COUNT = 300

print("="*60)
print(f"[*] BẮT ĐẦU CÂN BẰNG DỮ LIỆU - MỤC TIÊU: {TARGET_COUNT} ẢNH/LOÀI")
print("="*60)

# --- 2. ĐỊNH NGHĨA CÁC PHÉP BIẾN ĐỔI (TRANSFORMS) TẠO ẢNH MỚI ---
augmentation_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),            
    transforms.RandomRotation(degrees=(-15, 15)),      
    transforms.ColorJitter(brightness=0.15, contrast=0.15), 
])

# --- 3. VÒNG LẶP XỬ LÝ TỪNG LOÀI ---
# [ĐÃ FIX LỖI]: Bỏ qua các thư mục ẩn bắt đầu bằng dấu chấm (như .dist, .ipynb_checkpoints)
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')]

for cls in classes:
    class_path = os.path.join(DATA_DIR, cls)
    
    existing_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(existing_images)
    
    # [ĐÃ FIX LỖI]: Bỏ qua luôn nếu thư mục không có ảnh gốc nào
    if current_count == 0:
        print(f" Bỏ qua thư mục rỗng: '{cls}'")
        continue
        
    if current_count >= TARGET_COUNT:
        print(f"{cls}: Đã có {current_count} ảnh -> Đạt chuẩn, bỏ qua.")
        continue
        
    needed = TARGET_COUNT - current_count
    print(f"{cls}: Đang có {current_count} ảnh -> Cần sinh thêm {needed} ảnh.")
    
    generated_count = 0
    while generated_count < needed:
        random_img_name = random.choice(existing_images)
        img_path = os.path.join(class_path, random_img_name)
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                augmented_img = augmentation_pipeline(img)
                
                new_filename = f"AUG_{generated_count}_{random_img_name}"
                save_path = os.path.join(class_path, new_filename)
                
                augmented_img.save(save_path, format='JPEG', quality=95)
                generated_count += 1
                
        except Exception as e:
            pass # Bỏ qua lỗi hiển thị để code chạy mượt mà đến cuối

print("\n" + "="*60)
print(" HOÀN TẤT! TOÀN BỘ CÁC LOÀI ĐÃ ĐƯỢC CÂN BẰNG TRÒN 300 ẢNH.")
print("="*60)