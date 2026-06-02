# 👤 Thành viên 1: Đặng Thành Thi (Nhóm trưởng)
* **MSSV:** 2001230918
* **Vai trò:** Kỹ sư Dữ liệu & Phát triển Ứng dụng (Data Engineer & App Dev)
* **Đề tài phụ trách:** Thu thập, cân bằng dữ liệu đầu vào và phát triển giao diện ứng dụng nhận diện (Desktop & Web).

---

## 📂 Danh sách các file phụ trách:
1. **`crawl_bing_images.py`**: Script cào dữ liệu hình ảnh tự động từ Bing.
2. **`canbang.py`**: Cân bằng tập dữ liệu các loài động vật hoang dã bằng augmentation (lật ngang, xoay nhẹ, chỉnh sáng...) đạt mốc 300 ảnh/loài.
3. **`app_desktop.py`**: Ứng dụng Desktop chính thức viết bằng CustomTkinter tích hợp Grad-CAM giải thích quyết định của AI và bộ lọc ngưỡng tin cậy 85%.
4. **`appnhandien.py`**: Ứng dụng Web phụ bằng Streamlit cho phép người dùng upload ảnh/video nhận diện động vật trực tuyến.
5. **`Phan_1_Huong_Dan_Data_Pipeline.docx`**: Tài liệu Word chi tiết về Data Pipeline và Ứng dụng.

---

## 🛠️ Hướng dẫn chạy nhanh:
```bash
# Khởi chạy ứng dụng Desktop (CustomTkinter + Grad-CAM)
python app_desktop.py

# Khởi chạy ứng dụng Web (Streamlit)
streamlit run appnhandien.py
```
