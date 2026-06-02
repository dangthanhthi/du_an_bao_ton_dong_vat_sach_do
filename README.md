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

### 3. Trần Xuân Hướng (Thành viên)
* **MSSV:** 2001230339
* **Vai trò:** Kỹ sư Phân tích Dữ liệu & Đánh giá (Data Analyst & Evaluation)
* **Đề tài phụ trách:** Chia tập dữ liệu, chuẩn hóa ảnh, tính toán Loss/Accuracy và vẽ biểu đồ hiệu năng (ROC-AUC).

---

## 📂 Danh sách các file dự án chính:

### 🛠️ Mã nguồn & Ứng dụng (Đặng Thành Thi phụ trách)
1. **`crawl_bing_images.py`**: Script tự động thu thập và cào hình ảnh từ công cụ tìm kiếm Bing.
2. **`canbang.py`**: Script cân bằng số lượng hình ảnh của các lớp dữ liệu động vật bằng kỹ thuật Augmentation (lật ngang, xoay góc, hiệu chỉnh độ sáng/độ tương phản) đạt chuẩn 300 ảnh/loài.
3. **`app_desktop.py`**: Ứng dụng nhận diện trên máy tính (Desktop App) hoàn thiện, phát triển bằng thư viện CustomTkinter, tích hợp Grad-CAM giải thích trực quan hóa vùng mô hình tập trung và bộ lọc kiểm soát ngưỡng tin cậy >= 85%.
4. **`appnhandien.py`**: Ứng dụng Web nhận diện trực tuyến (Web App) phát triển bằng Streamlit, cho phép người dùng tải lên hình ảnh hoặc video để nhận diện đối tượng theo thời gian thực.

### 🧠 Cấu trúc & Huấn luyện mô hình AI (Lưu Đức Linh phụ trách)
5. **`resnet18finetuning.py`**: Thiết lập cấu trúc Fine-tuning ResNet-18 (khóa Layer 1 & 2, rã đông Layer 3 & 4) và đầu phân loại tùy chỉnh (`fc` head).
6. **`train_resnet18_full.py`**: Vòng lặp huấn luyện (Epoch Training Loop), luân chuyển giữa pha `train` và `val`, tự động lưu trọng số tốt nhất (`ResNet18_Best_Weights.pth`).

### 📊 Phân tách & Đánh giá Dữ liệu (Trần Xuân Hướng phụ trách)
7. **`tienxulyanh.py`**: Tiền xử lý và chia bộ dữ liệu theo tỷ lệ chuẩn 7:2:1 (Train: 70%, Val: 20%, Test: 10%), dọn dẹp các thư mục rác hệ thống.
8. **`phantichdulieu.py`**: Phân tích phân phối và biểu đồ của bộ dữ liệu đầu vào.
9. **`phantichthongso.py`**: Tính toán các thông số huấn luyện và đánh giá mô hình.
10. **`run_video_inference.py`**: Giao diện phụ bằng Tkinter để kiểm tra suy luận luồng video và ảnh nhanh chóng.

### 💾 Trọng số & Kết quả phân tích (Chung)
11. **`ResNet18_Best_Weights.pth`**: Trọng số tối ưu nhất được lưu của mô hình ResNet-18 sau quá trình huấn luyện.
12. **`Metadata_Dataset_Final.xlsx`**: Bảng dữ liệu Excel quản lý siêu dữ liệu cuối cùng của tập dữ liệu.
13. **`Loss_Accuracy_Curves.png`**: Biểu đồ hiển thị biến thiên của tổn thất (Loss) và độ chính xác (Accuracy) qua các epoch.
14. **`ROC_AUC_Curve.png`**: Biểu đồ ROC-AUC đánh giá chất lượng phân loại đa lớp của mô hình.

---

## 🛠️ Hướng dẫn cài đặt & Khởi chạy nhanh:

### 1. Cài đặt thư viện phụ thuộc
Đảm bảo bạn đã cài đặt đầy đủ các thư viện cần thiết trước khi chạy:
```bash
pip install torch torchvision numpy opencv-python pillow beautifulsoup4 requests customtkinter streamlit openpyxl pandas matplotlib scikit-learn
```

### 2. Thực hiện chia tập dữ liệu 7:2:1 (Tiền xử lý)
```bash
python tienxulyanh.py
```

### 3. Huấn luyện mô hình từ đầu (ResNet-18)
```bash
python train_resnet18_full.py
```

### 4. Khởi chạy ứng dụng Desktop (CustomTkinter + Grad-CAM)
```bash
python app_desktop.py
```

### 5. Khởi chạy ứng dụng Web (Streamlit)
```bash
streamlit run appnhandien.py
```

### 6. Khởi chạy ứng dụng kiểm thử video phụ
```bash
python run_video_inference.py
```
