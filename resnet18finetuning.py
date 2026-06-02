import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

print("="*65)
print("[*] BƯỚC 3: KHỞI TẠO VÀ TINH CHỈNH KIẾN TRÚC RESNET-18")
print("="*65)

# --- 1. THIẾT LẬP THIẾT BỊ TÍNH TOÁN (CPU hoặc GPU) ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[*] Mô hình sẽ được huấn luyện trên thiết bị: {device}")

# --- 2. TẢI MÔ HÌNH PRE-TRAINED TỪ IMAGENET ---
# Lấy trọng số (weights) tốt nhất và mới nhất từ PyTorch
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# --- 3. ĐÓNG BĂNG VÀ RÃ ĐÔNG TỪNG PHẦN (PARTIAL UNFREEZING) ---
# Bước 3.1: Đóng băng toàn bộ mạng (Không cho cập nhật trọng số)
for param in model.parameters():
    param.requires_grad = False

# Bước 3.2: Chỉ rã đông layer3 và layer4 để mạng học sâu hơn vào đặc trưng động vật
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

print("[*] Đã khóa các lớp Tích chập đầu tiên. Đã rã đông Layer 3 và Layer 4.")

# --- 4. TÙY CHỈNH LỚP PHÂN LOẠI CUỐI CÙNG (CUSTOM CLASSIFIER) ---
# Số lượng nơ-ron đầu vào của lớp cuối cùng trong ResNet-18 mặc định là 512
num_ftrs = model.fc.in_features
NUM_CLASSES = 9 # Dự án của bạn có 9 loài động vật

# Thay thế lớp fc cũ bằng một chuỗi các lớp mới để chống Overfitting
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5), # Tắt ngẫu nhiên 50% nơ-ron khi huấn luyện
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(device)
print(f"[*] Đã thay thế lớp phân loại cuối cùng cho {NUM_CLASSES} loài động vật.")

# --- 5. THIẾT LẬP HÀM MẤT MÁT (LOSS FUNCTION) ---
# Sử dụng Label Smoothing = 0.1 để giúp mô hình không quá tự tin, giảm thiểu sai số
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --- 6. THIẾT LẬP THUẬT TOÁN TỐI ƯU KÈM TỐC ĐỘ HỌC PHÂN CẤP ---
# Differential Learning Rates: Lớp cũ học chậm, lớp mới học nhanh
optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5}, # Tốc độ học rất nhỏ
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3}      # Tốc độ học tiêu chuẩn
], weight_decay=1e-4) # Thêm weight_decay để chuẩn hóa L2 (chống học vẹt)

print("[*] Đã khởi tạo Hàm mất mát (CrossEntropy) và Thuật toán tối ưu (Adam).")
print("\nKIẾN TRÚC MÔ HÌNH ĐÃ SẴN SÀNG CHO QUÁ TRÌNH HUẤN LUYỆN!")

# In ra tổng số tham số cần huấn luyện để kiểm tra
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[*] Tổng số tham số đang được huấn luyện: {trainable_params:,}")
print("="*65)