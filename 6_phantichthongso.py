import copy
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

print("="*65)
print("[*] BƯỚC 4: HUẤN LUYỆN MÔ HÌNH VÀ VẼ BIỂU ĐỒ ĐÁNH GIÁ")
print("="*65)

# --- 1. HÀM HUẤN LUYỆN VÀ THU THẬP LỊCH SỬ ---
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=15):
    since = time.time()
    
    # Các danh sách để lưu lại chỉ số vẽ biểu đồ
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Mỗi epoch có 2 pha: Huấn luyện (train) và Đánh giá (val)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Đặt mô hình ở chế độ học
            else:
                model.eval()   # Đặt mô hình ở chế độ đánh giá (tắt Dropout)

            running_loss = 0.0
            running_corrects = 0

            # Lấy từng lô dữ liệu
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # Xóa bộ nhớ đạo hàm cũ

                # Tính toán song song
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Cập nhật trọng số nếu đang ở pha train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Cộng dồn loss và độ chính xác
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Tính trung bình cho toàn bộ epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Lưu vào lịch sử để vẽ biểu đồ
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Lưu lại bộ trọng số tốt nhất dựa trên tập Validation
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nHuấn luyện hoàn tất trong {time_elapsed // 60:.0f}p {time_elapsed % 60:.0f}s')
    print(f'Độ chính xác Validation cao nhất: {best_acc:4f}')

    # Nạp lại bộ trọng số tốt nhất vào mô hình
    model.load_state_dict(best_model_wts)
    return model, history

# --- THỰC THI HUẤN LUYỆN ---
# (Giả định bạn đã nối tiếp các biến model, criterion, optimizer, dataloaders từ file trước)
# Nếu máy tính không có GPU mạnh, bạn có thể chỉnh num_epochs = 10 hoặc 15 để chạy cho nhanh
EPOCHS = 15
best_model, metrics_history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=EPOCHS)

# Lưu mô hình ra file để mang đi báo cáo
torch.save(best_model.state_dict(), 'ResNet18_Best_Weights.pth')
print("[*] Đã lưu mô hình thành công vào file 'ResNet18_Best_Weights.pth'")

# --- 2. VẼ BIỂU ĐỒ ACCURACY VÀ LOSS ---
print("[*] Đang xuất biểu đồ Accuracy và Loss...")
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(14, 5))

# Biểu đồ Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, metrics_history['train_loss'], label='Train Loss', color='blue', marker='o')
plt.plot(epochs_range, metrics_history['val_loss'], label='Validation Loss', color='red', marker='x')
plt.title('Biểu đồ Mất mát (Loss) qua các Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Biểu đồ Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, metrics_history['train_acc'], label='Train Accuracy', color='blue', marker='o')
plt.plot(epochs_range, metrics_history['val_acc'], label='Validation Accuracy', color='red', marker='x')
plt.title('Biểu đồ Độ chính xác (Accuracy) qua các Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('Loss_Accuracy_Curves.png')
plt.show()

# --- 3. VẼ BIỂU ĐỒ ROC VÀ TÍNH CHỈ SỐ AUC ĐA LỚP ---
print("[*] Đang tính toán ma trận xác suất và vẽ biểu đồ ROC/AUC...")

def plot_multiclass_roc(model, dataloader, class_names, num_classes, device):
    model.eval()
    y_true = []
    y_scores = []
    
    # Chạy tập Test/Val để lấy xác suất dự đoán
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Áp dụng Softmax để chuyển Output thành dạng phần trăm xác suất (0 đến 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            y_scores.extend(probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Chuyển nhãn thật thành dạng nhị phân ma trận (One-hot encoding) cho 9 lớp
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Tính toán ROC và AUC cho từng lớp
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2, 
                 label=f'ROC {class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
    # Vẽ đường trung bình ngẫu nhiên (Baseline)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline ngẫu nhiên (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính Giả (False Positive Rate)', fontsize=12)
    plt.ylabel('Tỷ lệ Dương tính Thật (True Positive Rate)', fontsize=12)
    plt.title('Biểu đồ ROC đa lớp (Multi-class ROC Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    
    plt.savefig('ROC_AUC_Curve.png')
    plt.show()

# Gọi hàm vẽ AUC (Sử dụng tập Test hoặc Val)
plot_multiclass_roc(best_model, dataloaders['val'], class_names, NUM_CLASSES, device)