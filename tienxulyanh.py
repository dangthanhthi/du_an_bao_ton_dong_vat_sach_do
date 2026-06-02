import os
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import splitfolders

print("="*65)
print("BƯỚC 2: TIỀN XỬ LÝ ẢNH & TẠO DATALOADER (TỈ LỆ 7:2:1)")
print("="*65)

# --- 1. ĐƯỜNG DẪN DỮ LIỆU ---
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
DATA_DIR = os.path.join(desktop_path, "Endangered_Animals_300_Unique")
SPLIT_DIR = os.path.join(desktop_path, "Dataset_ResNet18_Ready")

# --- 2. DỌN DẸP THƯ MỤC RÁC & CÀI ĐẶT LẠI ---
# Quét và tiêu diệt các thư mục ẩn (như .dist, .ipynb_checkpoints) trong thư mục gốc
for item in os.listdir(DATA_DIR):
    if item.startswith('.'):
        hidden_dir = os.path.join(DATA_DIR, item)
        if os.path.isdir(hidden_dir):
            shutil.rmtree(hidden_dir)
            print(f"[*] Đã dọn dẹp thư mục rác hệ thống: {item}")

# Xóa thư mục chia tỉ lệ cũ (nếu có) để chia lại theo chuẩn 7-2-1
if os.path.exists(SPLIT_DIR):
    print("[*] Đang làm sạch thư mục cũ để chia lại dữ liệu...")
    shutil.rmtree(SPLIT_DIR)

# --- 3. CHIA TẬP DỮ LIỆU (70% Train - 20% Valid - 10% Test) ---
print(f"[*] Đang tiến hành chia dữ liệu (Train: 70%, Val: 20%, Test: 10%)...")
splitfolders.ratio(DATA_DIR, output=SPLIT_DIR, seed=42, ratio=(0.7, 0.2, 0.1), group_prefix=None)
print(" Đã chia xong!")

# --- 4. ĐỊNH NGHĨA PHÉP BIẾN ĐỔI (TRANSFORMS) CHUẨN RESNET-18 ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(),                   
        transforms.ToTensor(),                               
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),                          
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Thêm tập Test: Biến đổi y hệt Val vì khi thi không được xoay lật ảnh gian lận
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),                          
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 5. KHỞI TẠO DATALOADER ---
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

print("\nTÓM TẮT DỮ LIỆU SAU KHI XỬ LÝ:")
print(f"  + Số lượng lớp (Classes): {len(class_names)}")
print(f"  + Tên các loài: {class_names}")
print(f"  + Tập Huấn luyện (Train): {dataset_sizes['train']} ảnh")
print(f"  + Tập Kiểm tra (Valid): {dataset_sizes['val']} ảnh")
print(f"  + Tập Đánh giá (Test): {dataset_sizes['test']} ảnh")
print("\n HOÀN TẤT CÔNG ĐOẠN 2! BỘ NÃO RESNET-18 ĐÃ SẴN SÀNG.")