import os
import shutil
import sys
from PIL import Image

# Reconfigure stdout for UTF-8 on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def clean_and_reset_dataset():
    print("="*60)
    print("[*] ĐANG TIẾN HÀNH DỌN DẸP & KHÔI PHỤC TẬP DỮ LIỆU GỐC")
    print("="*60)
    
    # 1. Đường dẫn dataset
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(workspace_dir, "Endangered_Animals_300_Unique")
    
    if not os.path.exists(DATA_DIR):
        print(f"[!] Thư mục dữ liệu '{DATA_DIR}' không tồn tại. Vui lòng kiểm tra lại!")
        return
        
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')]
    
    removed_aug_count = 0
    removed_corrupted_count = 0
    
    # Danh sách các tệp bị xác định là lỗi hoặc cần xóa đặc biệt
    corrupted_filenames = [
        "clouded-leopard-neofelis-nebulosa-art-wolfe.jpg"  # Lỗi verify ở lớp Bao_Gam
    ]
    
    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)
        files = os.listdir(class_path)
        
        print(f"\n[*] Đang xử lý lớp: {cls}")
        
        for f in files:
            file_path = os.path.join(class_path, f)
            
            # Chỉ xử lý file ảnh
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # 2. Xóa các tệp ảnh augment (từ canbang.py cũ)
            if f.startswith("AUG_"):
                try:
                    os.remove(file_path)
                    removed_aug_count += 1
                except Exception as e:
                    print(f"    [!] Lỗi khi xóa ảnh augment {f}: {e}")
                continue
                
            # 3. Xóa các tệp ảnh bị lỗi cấu trúc/không thể mở
            if f in corrupted_filenames:
                try:
                    os.remove(file_path)
                    removed_corrupted_count += 1
                    print(f"    [✓] Đã xóa ảnh lỗi đặc biệt: {f}")
                except Exception as e:
                    print(f"    [!] Không thể xóa {f}: {e}")
                continue
                
            # Kiểm tra xem ảnh có mở được bằng PIL không
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                # Nếu không verify được thì tiến hành xóa
                try:
                    os.remove(file_path)
                    removed_corrupted_count += 1
                    print(f"    [✓] Đã xóa ảnh bị hỏng cấu trúc: {f} (Lỗi: {e})")
                except Exception as ex:
                    print(f"    [!] Không thể xóa ảnh hỏng {f}: {ex}")
                    
    print("\n" + "="*60)
    print("HOÀN TẤT DỌN DẸP:")
    print(f"  + Số ảnh Augment (AUG_...) đã xóa: {removed_aug_count}")
    print(f"  + Số ảnh bị lỗi cấu trúc đã xóa: {removed_corrupted_count}")
    print(" Tập dữ liệu gốc đã được đưa về trạng thái tinh khiết!")
    print("="*60)

if __name__ == "__main__":
    clean_and_reset_dataset()
