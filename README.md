# 🦁 Dự Án Nhận Diện 9 Loài Động Vật Sách Đỏ Việt Nam

Dự án phát triển mô hình Deep Learning dựa trên kiến trúc **ResNet-18 Transfer Learning** nhằm nhận diện và phân loại 9 loài động vật nguy cấp có tên trong Sách Đỏ. Dự án phục vụ môn học Deep Learning.

---

## 👤 Thành viên thực hiện:

### 1. Đặng Thành Thi (Nhóm trưởng)
* **MSSV:** 2001230918
* **Vai trò:** Kỹ sư Dữ liệu & Phát triển Ứng dụng (Data Engineer & App Dev)
* **Đề tài phụ trách:** Thu thập, cân bằng dữ liệu đầu vào và phát triển giao diện ứng dụng nhận diện (Desktop & Web).

### 2. Lưu Đức Linh (Thành viên)
* **MSSV:** 2001230444
* **Vai trò:** Nhà nghiên cứu Mô hình AI (AI Researcher & Training Specialist)
* **Đề tài phụ trách:** Thiết kế kiến trúc mô hình ResNet-18 Transfer Learning và lập trình vòng lặp huấn luyện chính.

---

## 📂 Danh sách các file dự án chính:
1. **`crawl_bing_images.py`**: Script tự động thu thập và cào hình ảnh từ công cụ tìm kiếm Bing.
2. **`canbang.py`**: Script cân bằng số lượng hình ảnh của các lớp dữ liệu động vật bằng kỹ thuật Augmentation (lật ngang, xoay góc, hiệu chỉnh độ sáng/độ tương phản) đạt chuẩn 300 ảnh/loài.
3. **`resnet18finetuning.py`**: Thiết lập cấu trúc Fine-tuning ResNet-18 (khóa Layer 1 & 2, rã đông Layer 3 & 4) và đầu phân loại tùy chỉnh (`fc` head).
4. **`train_resnet18_full.py`**: Vòng lặp huấn luyện (Epoch Training Loop), luân chuyển giữa pha `train` và `val`, tự động lưu trọng số tốt nhất (`ResNet18_Best_Weights.pth`).
5. **`app_desktop.py`**: Ứng dụng nhận diện trên máy tính (Desktop App) hoàn thiện, phát triển bằng thư viện CustomTkinter, tích hợp Grad-CAM giải thích trực quan hóa vùng mô hình tập trung và bộ lọc kiểm soát ngưỡng tin cậy >= 85%.
6. **`appnhandien.py`**: Ứng dụng Web nhận diện trực tuyến (Web App) phát triển bằng Streamlit, cho phép người dùng tải lên hình ảnh hoặc video để nhận diện đối tượng theo thời gian thực.
7. **`Metadata_Dataset_Final.xlsx`**: Bảng dữ liệu Excel quản lý siêu dữ liệu cuối cùng của tập dữ liệu.
8. **`ResNet18_Best_Weights.pth`**: Trọng số tối ưu nhất được lưu của mô hình ResNet-18 sau quá trình huấn luyện.

---

## 🛠️ Hướng dẫn cài đặt & Khởi chạy nhanh:

### 1. Cài đặt thư viện phụ thuộc
Đảm bảo bạn đã cài đặt đầy đủ các thư viện cần thiết trước khi chạy:
```bash
pip install torch torchvision numpy opencv-python pillow beautifulsoup4 requests customtkinter streamlit openpyxl splitfolders scikit-learn matplotlib
```

### 2. Huấn luyện mô hình từ đầu (ResNet-18)
```bash
python train_resnet18_full.py
```

### 3. Khởi chạy ứng dụng Desktop (CustomTkinter + Grad-CAM)
```bash
python app_desktop.py
```

### 4. Khởi chạy ứng dụng Web (Streamlit)
```bash
streamlit run appnhandien.py
```
