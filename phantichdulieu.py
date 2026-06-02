import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. THIẾT LẬP ĐƯỜNG DẪN ---
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
DATA_DIR = os.path.join(desktop_path, "Endangered_Animals_300_Unique")

print("="*50)
print("[*] BƯỚC 1: PHÂN TÍCH DỮ LIỆU KHÁM PHÁ (EDA)")
print("="*50)

# --- 2. ĐẾM SỐ LƯỢNG ẢNH & VẼ BIỂU ĐỒ ---
# [ĐÃ SỬA LỖI] Yêu cầu chỉ quét Thư mục, bỏ qua file Excel
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
class_counts = {}

for cls in classes:
    class_path = os.path.join(DATA_DIR, cls)
    num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    class_counts[cls] = num_images

# Vẽ biểu đồ cột
plt.figure(figsize=(12, 6))
bars = plt.bar(class_counts.keys(), class_counts.values(), color='skyblue', edgecolor='black')
plt.title('Phân phối Dữ liệu 9 Loài Động Vật Hoang Dã', fontsize=16, fontweight='bold')
plt.xlabel('Tên Loài', fontsize=12)
plt.ylabel('Số lượng ảnh', fontsize=12)
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(desktop_path, 'Bieu_Do_Du_Lieu.png')) 
plt.show()

# --- 3. HIỂN THỊ LƯỚI ẢNH MẪU (IMAGE GRID) ---
print("\n[*] Đang hiển thị lưới ảnh mẫu ngẫu nhiên...")
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle('Ảnh Mẫu Từ Tập Dữ Liệu', fontsize=16, fontweight='bold')

for i, cls in enumerate(classes[:9]):
    class_path = os.path.join(DATA_DIR, cls)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if images:
        random_img_name = random.choice(images)
        img_path = os.path.join(class_path, random_img_name)
        img = Image.open(img_path)
        
        ax = axes[i//3, i%3]
        ax.imshow(img)
        ax.set_title(cls, fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(desktop_path, 'Luoi_Anh_Mau.png'))
plt.show()