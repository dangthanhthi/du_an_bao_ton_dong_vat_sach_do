import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import splitfolders
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=15):
    since = time.time()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nHuan luyen hoan tat trong {time_elapsed // 60:.0f}p {time_elapsed % 60:.0f}s')
    print(f'Do chinh xac Validation cao nhat: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

def plot_multiclass_roc(model, dataloader, class_names, num_classes, device):
    print("[*] Dang tinh toan ma tran xac suat va ve bieu do ROC/AUC...")
    model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            y_scores.extend(probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
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
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    
    plt.savefig('ROC_AUC_Curve_New.png')
    plt.show()

# KHOI LENH BAO VE MULTIPROCESSING TREN WINDOWS
if __name__ == '__main__':
    # Reconfigure stdout for UTF-8 on Windows terminal
    import sys
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    print("="*65)
    print("[*] KHỞI ĐỘNG HỆ THỐNG HUẤN LUYỆN RESNET-18 (FIX DATA LEAKAGE)")
    print("="*65)

    # 1. THIẾT LẬP ĐƯỜNG DẪN VÀ CHẠY TIỀN XỬ LÝ NẾU CHƯA CÓ
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    DATA_DIR = os.path.join(desktop_path, "Endangered_Animals_300_Unique")
    SPLIT_DIR = os.path.join(desktop_path, "Dataset_ResNet18_Ready")

    # Nếu chưa chia dữ liệu hoặc thư mục trống, chạy kịch bản tienxulyanh.py
    if not os.path.exists(SPLIT_DIR) or not os.listdir(SPLIT_DIR):
        print("[*] Thư mục dữ liệu chia sẵn không tồn tại. Đang chạy 'step3_tienxulyanh.py' để tiền xử lý và cân bằng tập Train...")
        import subprocess
        subprocess.run(["python", "step3_tienxulyanh.py"], check=True)
    
    # 2. CẤU HÌNH DATALOADER & BIẾN ĐỔI ẢNH (TRANSFORMS)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),                       # Xoay ngẫu nhiên
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Đổi màu sắc ngẫu nhiên
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

    BATCH_SIZE = 32
    image_datasets = {x: datasets.ImageFolder(os.path.join(SPLIT_DIR, x), data_transforms[x]) 
                      for x in ['train', 'val', 'test']}
    
    # Khởi tạo Dataloader
    # LƯU Ý HỌC THUẬT CHO BÁO CÁO: 
    # Nếu muốn dùng WeightedRandomSampler cân bằng trực tiếp trong RAM thay vì cân bằng trên đĩa cứng:
    # --------------------------------------------------------------------------------------
    # train_dataset = image_datasets['train']
    # class_counts = [0] * len(train_dataset.classes)
    # for _, label in train_dataset.samples:
    #     class_counts[label] += 1
    # class_weights = 1.0 / np.array(class_counts)
    # sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    #
    # Khi đó ở dict dataloaders dưới đây, phần 'train' cấu hình như sau:
    # 'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    # (Bỏ shuffle=True vì shuffle và sampler không thể dùng chung)
    # --------------------------------------------------------------------------------------

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=2) 
                   for x in ['train', 'val', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    NUM_CLASSES = len(class_names)

    # 3. KHỞI TẠO MÔ HÌNH RESNET-18
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị huấn luyện: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, NUM_CLASSES)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-5},
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    # 4. THỰC THI HUẤN LUYỆN
    EPOCHS = 15
    print("\n[*] BẮT ĐẦU HUAN LUYEN...")
    best_model, metrics_history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=EPOCHS)

    torch.save(best_model.state_dict(), 'ResNet18_Best_Weights.pth')
    print("[*] Đã lưu mô hình tốt nhất vào file 'ResNet18_Best_Weights.pth'")

    # 5. VE BIEU DO
    print("[*] Dang xuat bieu do Accuracy va Loss...")
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, metrics_history['train_loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(epochs_range, metrics_history['val_loss'], label='Validation Loss', color='red', marker='x')
    plt.title('Loss qua cac Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, metrics_history['train_acc'], label='Train Accuracy', color='blue', marker='o')
    plt.plot(epochs_range, metrics_history['val_acc'], label='Validation Accuracy', color='red', marker='x')
    plt.title('Accuracy qua cac Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('Loss_Accuracy_Curves_New.png')
    plt.show()

    plot_multiclass_roc(best_model, dataloaders['val'], class_names, NUM_CLASSES, device)