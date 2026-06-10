import os
import sys
import re
import json
import urllib.request
import requests
from bs4 import BeautifulSoup

def download_bing_images(query, limit=100, output_dir="raw_data"):
    print(f"[*] Đang cào ảnh từ Bing cho từ khóa: '{query}'...")
    
    # Tạo thư mục lưu trữ nếu chưa có
    save_dir = os.path.join(output_dir, query.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)
    
    # Cấu hình header để giả lập trình duyệt tránh bị Bing chặn
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}&first=1"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Tìm tất các các thẻ chứa liên kết ảnh gốc trong Bing
        image_elements = soup.find_all("a", class_="iusc")
        
        links = []
        for el in image_elements:
            try:
                m_attr = el.get("m")
                m_json = json.loads(m_attr)
                links.append(m_json["murl"]) # murl chứa link ảnh gốc
            except Exception:
                continue
                
        print(f"[*] Tìm thấy {len(links)} liên kết ảnh.")
        
        count = 0
        for i, link in enumerate(links):
            if count >= limit:
                break
            try:
                # Tải ảnh về
                img_data = requests.get(link, headers=headers, timeout=5).content
                # Lấy phần mở rộng
                ext = ".jpg"
                if ".png" in link.lower():
                    ext = ".png"
                elif ".jpeg" in link.lower():
                    ext = ".jpeg"
                    
                file_name = f"img_{count}{ext}"
                file_path = os.path.join(save_dir, file_name)
                
                with open(file_path, "wb") as f:
                    f.write(img_data)
                
                count += 1
                print(f" -> Đã tải ({count}/{limit}): {file_name}")
            except Exception as e:
                # Bỏ qua nếu link hỏng hoặc timeout
                continue
                
        print(f"[+] Hoàn tất! Đã tải thành công {count} ảnh lưu tại '{save_dir}'.")
    except Exception as e:
        print(f"[!] Lỗi khi cào ảnh: {e}")

if __name__ == "__main__":
    # Ví dụ chạy cào thử ảnh
    # download_bing_images("Vooc Cha Va", limit=10)
    print("[*] Script cào ảnh tự động từ Bing đã sẵn sàng.")
    print("Để chạy cào ảnh, gọi hàm: download_bing_images('Tên loài', limit=200)")
