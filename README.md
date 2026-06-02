# 🦁 Dự Án Nhận Diện 9 Loài Động Vật Sách Đỏ Việt Nam

Dự án nghiên cứu và phát triển mô hình Deep Learning dựa trên phương pháp **Transfer Learning** với kiến trúc **ResNet-18** để nhận diện, phân loại và trực quan hóa vùng chú ý của AI đối với 9 loài động vật quý hiếm có tên trong Sách Đỏ. 

Dự án này được thiết kế theo dạng **Pipeline khép kín (End-to-End Deep Learning Pipeline)**, giúp các thành viên trong nhóm dễ dàng nắm bắt toàn bộ quy trình từ xử lý dữ liệu thô cho đến sản phẩm ứng dụng thực tế.

---

## 🗺️ Sơ Đồ Quy Trình Pipeline Hệ Thống (Workflow)

Dưới đây là luồng đi của dữ liệu và các bước phát triển trong dự án. Tất cả 3 thành viên đều có thể nhìn vào sơ đồ này để hiểu cách các file mã nguồn liên kết với nhau:

```mermaid
graph TD
    %% Định nghĩa các bước trong Pipeline
    subgraph Phase 1: Kỹ Thuật Dữ Liệu
        A[Thu thập dữ liệu thô<br><b>crawl_bing_images.py</b>] --> B[Cân bằng dữ liệu & Augment<br><b>canbang.py</b>]
    end

    subgraph Phase 2: Tiền Xử Lý & Phân Tích
        B --> C[Phân chia tập 7:2:1<br><b>tienxulyanh.py</b>]
        C --> D[Phân tích phân phối dữ liệu<br><b>phantichdulieu.py</b>]
    end

    subgraph Phase 3: Thiết Kế & Huấn Luyện AI
        D --> E[Thiết lập cấu trúc ResNet-18<br><b>resnet18finetuning.py</b>]
        E --> F[Vòng lặp huấn luyện tối ưu<br><b>train_resnet18_full.py</b>]
    end

    subgraph Phase 4: Đánh Giá & Kiểm Thử
        F --> G[Lưu trọng số tốt nhất<br><b>ResNet18_Best_Weights.pth</b>]
        G --> H[Tính toán ROC-AUC / Loss / Acc<br><b>phantichthongso.py</b>]
    end

    subgraph Phase 5: Triển Khai Ứng Dụng
        G --> I[Ứng dụng Desktop + Grad-CAM<br><b>app_desktop.py</b>]
        G --> J[Ứng dụng Web Streamlit<br><b>appnhandien.py</b>]
        G --> K[Giao diện phụ video nhanh<br><b>run_video_inference.py</b>]
    end

    %% Định nghĩa màu sắc thẩm mỹ
    style A fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style B fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style C fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style D fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style E fill:#f96,stroke:#333,stroke-width:2px,color:#000
    style F fill:#f96,stroke:#333,stroke-width:2px,color:#000
    style G fill:#55ea82,stroke:#333,stroke-width:3px,color:#000
    style H fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style I fill:#fbc531,stroke:#333,stroke-width:2px,color:#000
    style J fill:#fbc531,stroke:#333,stroke-width:2px,color:#000
    style K fill:#fbc531,stroke:#333,stroke-width:2px,color:#000
```

---

## 👥 Phân Công Nhiệm Vụ & Vai Trò Thành Viên

### 1. Đặng Thành Thi (Nhóm trưởng - Data Engineer & App Developer)
* **MSSV:** 2001230918
* **Vai trò:** Kỹ sư Dữ liệu & Phát triển Ứng dụng.
* **Nhiệm vụ:**
  - Thiết kế pipeline thu thập ảnh tự động (`crawl_bing_images.py`).
  - Áp dụng các thuật toán Augmentation (lật, xoay, tăng giảm sáng...) để cân bằng dữ liệu đạt chuẩn 300 ảnh/loài (`canbang.py`).
  - Lập trình giao diện ứng dụng Desktop chuyên nghiệp tích hợp giải thích AI Grad-CAM (`app_desktop.py`) và ứng dụng Web trực tuyến Streamlit (`appnhandien.py`).

### 2. Lưu Đức Linh (Thành viên - AI Researcher & Training Specialist)
* **MSSV:** 2001230444
* **Vai trò:** Nhà nghiên cứu Mô hình AI & Huấn luyện.
* **Nhiệm vụ:**
  - Thiết kế kiến trúc Fine-tuning trên nền tảng ResNet-18: đóng băng (freeze) các tầng mạng dưới (Layer 1, 2) để giữ nguyên khả năng trích xuất đặc trưng cơ bản và rã đông (unfreeze) các tầng mạng trên (Layer 3, 4) để học đặc trưng riêng biệt của 9 loài động vật (`resnet18finetuning.py`).
  - Thiết lập vòng lặp huấn luyện chính xác, cơ chế lưu mô hình tốt nhất (`train_resnet18_full.py`).

### 3. Trần Xuân Hướng (Thành viên - Data Analyst & Evaluation Specialist)
* **MSSV:** 2001230339
* **Vai trò:** Kỹ sư Phân tích Dữ liệu & Đánh giá Hiệu năng.
* **Nhiệm vụ:**
  - Phân chia tập dữ liệu huấn luyện, kiểm thử theo tỷ lệ khoa học 7:2:1 (Train, Val, Test) (`tienxulyanh.py`).
  - Thống kê biểu đồ phân bố và chất lượng tập dữ liệu đầu vào (`phantichdulieu.py`).
  - Tính toán các chỉ số đánh giá chuyên sâu (Loss, Accuracy, Confusion Matrix, ROC-AUC) để chứng minh tính chính xác của mô hình (`phantichthongso.py`).
  - Xây dựng giao diện phụ kiểm thử nhanh bằng Tkinter (`run_video_inference.py`).

---

## 📖 Chi Tiết Toàn Bộ Quá Trình Pipeline (5 Giai Đoạn)

Để cả 3 thành viên đều hiểu rõ cách hệ thống vận hành từ đầu đến cuối, dưới đây là mô tả chi tiết của từng giai đoạn:

### Giai Đoạn 1: Chuẩn Bị & Cân Bằng Dữ Liệu (Data Engineering)
* **Mục tiêu:** Xây dựng tập dữ liệu đồng đều, chất lượng cao để mô hình AI học tốt nhất, tránh hiện tượng lệch lớp (class imbalance) khiến AI chỉ nhận diện tốt một số loài nhất định.
* **Cách hoạt động:**
  1. Cào ảnh tự động từ Bing (`crawl_bing_images.py`) để gom hình ảnh gốc.
  2. Phát hiện các thư mục thiếu ảnh và tự động áp dụng kỹ thuật **Image Augmentation** (`canbang.py`) bao gồm: xoay ngẫu nhiên từ -15 đến 15 độ, lật ngang ảnh, và hiệu chỉnh độ sáng/độ tương phản để nhân bản dữ liệu một cách tự nhiên cho đến khi **mỗi loài có đủ chính xác 300 hình ảnh**.

### Giai Đoạn 2: Tiền Xử Lý & Phân Tích Phân Phối (Data Analysis)
* **Mục tiêu:** Chuẩn bị dữ liệu sẵn sàng cho huấn luyện và phân tích mức độ đa dạng của dữ liệu.
* **Cách hoạt động:**
  1. Chia tập dữ liệu (`tienxulyanh.py`) theo tỷ lệ chuẩn **70% để học (Train)**, **20% để tinh chỉnh (Validation)**, và **10% để kiểm tra độc lập (Test)**.
  2. Phân tích phân phối ảnh (`phantichdulieu.py`) để kiểm tra tính đa dạng trước khi đưa vào mô hình huấn luyện.

### Giai Đoạn 3: Thiết Kế Mô Hình & Huấn Luyện (Model Training)
* **Mục tiêu:** Huấn luyện mạng nơ-ron sâu nhận diện chính xác 9 loài động vật.
* **Cách hoạt động:**
  1. Load kiến trúc mạng **ResNet-18** đã được tiền huấn luyện trên tập dữ liệu khổng lồ ImageNet (`resnet18finetuning.py`). 
  2. Thực hiện **Fine-tuning**: Đóng băng các layer đầu (giữ nguyên khả năng nhận diện hình học, màu sắc cơ bản) và thay thế lớp Fully Connected (FC) cuối cùng thành một bộ phân loại mới gồm 9 lớp tương ứng với 9 loài động vật sách đỏ.
  3. Huấn luyện mô hình (`train_resnet18_full.py`): Sử dụng hàm mất mát `CrossEntropyLoss` và bộ tối ưu hóa `Adam`. Qua mỗi Epoch (vòng lặp học), mô hình tự động kiểm tra sai số trên tập Validation và chỉ lưu lại tệp trọng số tốt nhất là **`ResNet18_Best_Weights.pth`** để đưa vào ứng dụng thực tế.

### Giai Đoạn 4: Đánh Giá Chất Lượng (Model Evaluation)
* **Mục tiêu:** Chứng minh độ chính xác và tính khoa học của mô hình bằng các số liệu toán học.
* **Cách hoạt động:**
  1. Sử dụng script `phantichthongso.py` chạy kiểm thử trên tập dữ liệu Test độc lập (AI chưa từng thấy lúc học).
  2. Vẽ biểu đồ biến thiên độ chính xác (`Loss_Accuracy_Curves.png`) để chứng minh mô hình hội tụ tốt, không bị hiện tượng học vẹt (overfitting).
  3. Vẽ biểu đồ `ROC_AUC_Curve.png` đánh giá tỷ lệ dương tính thật/giả của từng loài, giúp chứng minh hiệu năng phân loại đa lớp tối ưu.

### Giai Đoạn 5: Triển Khai Ứng Dụng Sản Phẩm (Deployment)
* **Mục tiêu:** Đưa mô hình AI đã huấn luyện vào giao diện thực tế để người dùng cuối sử dụng.
* **Cách hoạt động:**
  1. **Ứng dụng Desktop chuyên nghiệp (`app_desktop.py`):**
     - Viết bằng CustomTkinter tạo giao diện tối (Dark mode) hiện đại.
     - Tích hợp kỹ thuật **Grad-CAM**: Đọc kích hoạt (activations) và đạo hàm (gradients) từ lớp tích chập cuối cùng của mô hình ResNet-18 để vẽ bản đồ nhiệt trực quan hóa vùng trọng tâm trên ảnh mà AI đang "nhìn" để đưa ra quyết định.
     - Tích hợp bộ lọc ngưỡng tin cậy **85%**: Nếu mô hình dự đoán loài động vật dưới 85%, hệ thống sẽ cảnh báo là "loài lạ" để ngăn ngừa nhận diện sai.
  2. **Ứng dụng Web trực tuyến Streamlit (`appnhandien.py`):**
     - Cho phép người dùng tải lên hình ảnh hoặc chạy suy luận luồng video theo thời gian thực trên giao diện trình duyệt web cực kỳ mượt mà.

---

## 🛠️ Hướng Dẫn Cài Đặt & Khởi Chạy Nhanh Cho Cả 3 Thành Viên

Mở Terminal tích hợp trong **VS Code** (Phím tắt `Ctrl + \``) và chạy các lệnh dưới đây tùy theo mục đích công việc của mình:

### Bước 1: Cài đặt toàn bộ thư viện cần thiết
```bash
pip install torch torchvision numpy opencv-python pillow beautifulsoup4 requests customtkinter streamlit openpyxl pandas matplotlib scikit-learn
```

### Bước 2: Chạy Tiền xử lý (Chia tập dữ liệu)
```bash
python tienxulyanh.py
```

### Bước 3: Chạy Cân bằng dữ liệu (Nếu cần thêm ảnh tự động)
```bash
python canbang.py
```

### Bước 4: Chạy huấn luyện mô hình (Học từ đầu)
```bash
python train_resnet18_full.py
```

### Bước 5: Khởi chạy Ứng dụng Desktop chính thức (CustomTkinter + Grad-CAM)
```bash
python app_desktop.py
```

### Bước 6: Khởi chạy Ứng dụng Web trực tuyến (Streamlit)
```bash
streamlit run appnhandien.py
```

### Bước 7: Khởi chạy giao diện phụ video kiểm thử nhanh
```bash
python run_video_inference.py
```
