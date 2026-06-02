import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk, ImageFile, ImageDraw, ImageFont
import os
import customtkinter as ctk
import numpy as np
from datetime import datetime
import time

# Hỗ trợ đọc ảnh bị lỗi cấu trúc
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Cấu hình CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- DATA: THÔNG TIN SÁCH ĐỎ VIỆT NAM ---
ANIMAL_INFO = {
    'Báo Gấm': {'status': 'Nguy cấp (EN)', 'desc': 'Loài mèo lớn có hoa văn đám mây độc đáo. Sống trong rừng sâu.'},
    'Cu Li Nhỏ': {'status': 'Nguy cấp (EN)', 'desc': 'Linh trưởng nhỏ, mắt to, hoạt động về đêm. Di chuyển chậm.'},
    'Gấu Ngựa': {'status': 'Nguy cấp (EN)', 'desc': 'Bộ lông đen với dải chữ V trắng đặc trưng trước ngực.'},
    'Hổ Đông Dương': {'status': 'Cực kỳ nguy cấp (CR)', 'desc': 'Chúa tể sơn lâm. Biểu tượng của sức mạnh hoang dã.'},
    'Hươu Sao': {'status': 'Cực kỳ nguy cấp (CR)', 'desc': 'Loài hươu có các đốm trắng như sao trên lưng.'},
    'Tê Tê Java': {'status': 'Cực kỳ nguy cấp (CR)', 'desc': 'Động vật có vảy duy nhất. Cuộn tròn khi gặp nguy hiểm.'},
    'Voi Châu Á': {'status': 'Nguy cấp (EN)', 'desc': 'Động vật lớn nhất VN. Có tính xã hội và trí tuệ cao.'},
    'Voọc Chà Vá': {'status': 'Nguy cấp (EN)', 'desc': 'Được mệnh danh là "Nữ hoàng linh trưởng" vì màu sắc rực rỡ.'},
    'Vượn Má Vàng': {'status': 'Nguy cấp (EN)', 'desc': 'Tiếng hót vang vọng khắp rừng già vào buổi sáng.'}
}

CLASS_NAMES = list(ANIMAL_INFO.keys())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. MÔ HÌNH VÀ GRAD-CAM LOGIC ---
class GradCAMModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features(x)
        h = x.register_hook(self.activations_hook)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features(x)

def load_full_model():
    base_model = models.resnet18(weights=None)
    base_model.fc = nn.Sequential(
        nn.Linear(base_model.fc.in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, len(CLASS_NAMES))
    )
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, 'ResNet18_Best_Weights.pth')
    if os.path.exists(weights_path):
        base_model.load_state_dict(torch.load(weights_path, map_location=device))
    base_model.eval()
    return base_model.to(device)

raw_model = load_full_model()
cam_model = GradCAMModel(raw_model)

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_prediction_and_cam(pil_img):
    img_tensor = data_transform(pil_img).unsqueeze(0).to(device)
    out = cam_model(img_tensor)
    probs = torch.nn.functional.softmax(out, dim=1)[0]
    top3_prob, top3_idx = torch.topk(probs, 3)
    class_idx = top3_idx[0].item()
    out[:, class_idx].backward()
    gradients = cam_model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = cam_model.get_activations(img_tensor).detach()
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1
    top3 = [(CLASS_NAMES[top3_idx[i]], top3_prob[i].item()) for i in range(3)]
    return top3, heatmap.numpy()

# --- 2. GIAO DIỆN MASTERPIECE V3.5.2 ---
class AnimalApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hệ Thống Phân Tích Động Vật Hoang Dã - Pro v3.5.2")
        self.geometry("1400x850")
        self.cap = None
        self.video_running = False
        self.session_history = []
        self.prev_time = 0
        self.detection_stats = {}
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.btn_img = ctk.CTkButton(self.sidebar, text="📸 Nhận diện Ảnh", height=45, command=self.process_image)
        self.btn_img.pack(padx=20, pady=10)
        self.btn_vid = ctk.CTkButton(self.sidebar, text="🎥 Nhận diện Video", height=45, fg_color="#2a8c55", command=self.start_video)
        self.btn_vid.pack(padx=20, pady=10)
        self.btn_stop = ctk.CTkButton(self.sidebar, text="⏹ Dừng xử lý", height=40, fg_color="#a13333", state="disabled", command=self.stop_video)
        self.btn_stop.pack(padx=20, pady=10)
        self.theme_menu = ctk.CTkOptionMenu(self.sidebar, values=["Dark", "Light"], command=ctk.set_appearance_mode)
        self.theme_menu.pack(padx=20, pady=20, side="bottom")
        self.center_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.center_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.center_frame.grid_columnconfigure(0, weight=1)
        self.center_frame.grid_rowconfigure(1, weight=1)
        self.lbl_status_bar = ctk.CTkLabel(self.center_frame, text="", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_status_bar.grid(row=0, column=0, pady=(0, 15))
        self.display_container = ctk.CTkFrame(self.center_frame, fg_color="#000", corner_radius=15)
        self.display_container.grid(row=1, column=0, sticky="nsew")
        self.main_display = ctk.CTkLabel(self.display_container, text="")
        self.main_display.place(relx=0.5, rely=0.5, anchor="center")
        self.analysis_panel = ctk.CTkFrame(self, width=350, corner_radius=15)
        self.analysis_panel.grid(row=0, column=2, padx=15, pady=15, sticky="nsew")
        ctk.CTkLabel(self.analysis_panel, text="GRAD-CAM", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        self.cam_display = ctk.CTkLabel(self.analysis_panel, text="Chờ phân tích...", width=280, height=200, fg_color="#1a1a1a", corner_radius=10)
        self.cam_display.pack(padx=20, pady=10)
        ctk.CTkLabel(self.analysis_panel, text="XÁC SUẤT PHÂN LOẠI", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        self.prob_bars = []
        for i in range(3):
            frame = ctk.CTkFrame(self.analysis_panel, fg_color="transparent")
            frame.pack(fill="x", padx=25, pady=5)
            lbl = ctk.CTkLabel(frame, text=f"---", font=ctk.CTkFont(size=12))
            lbl.pack(side="left")
            bar = ctk.CTkProgressBar(frame, width=150)
            bar.pack(side="right", padx=5)
            bar.set(0)
            self.prob_bars.append((lbl, bar))
        self.info_text = ctk.CTkTextbox(self.analysis_panel, height=200, font=ctk.CTkFont(size=12))
        self.info_text.pack(padx=20, pady=20, fill="x")
        self.info_text.insert("0.0", "Hồ sơ động vật sẽ hiển thị ở đây...")
        self.info_text.configure(state="disabled")

    def process_image(self):
        self.stop_video()
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not path: return
        img = Image.open(path).convert('RGB')
        self.analyze_and_show(img)

    def analyze_and_show(self, pil_img):
        start_t = time.time()
        top3, heatmap = get_prediction_and_cam(pil_img)
        latency = (time.time() - start_t) * 1000
        label, score = top3[0]
        
        # Threshold 0.85
        if score > 0.85:
            self.lbl_status_bar.configure(text=f"KẾT QUẢ: {label} ({score*100:.1f}%)", text_color="#4dbd74")
        else:
            self.lbl_status_bar.configure(text="LOÀI LẠ: Không nằm trong danh sách bảo tồn.", text_color="#e74c3c")
            
        self.show_image_on_label(pil_img, self.main_display, (800, 480))
        cam_img = self.apply_heatmap(pil_img, heatmap)
        self.show_image_on_label(cam_img, self.cam_display, (280, 180))
        for i, (name, prob) in enumerate(top3):
            self.prob_bars[i][0].configure(text=f"{name[:12]}")
            self.prob_bars[i][1].set(prob)
        self.update_info_card(label, score)

    def apply_heatmap(self, pil_img, heatmap):
        heatmap = cv2.resize(heatmap, pil_img.size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_np = np.array(pil_img)
        superimposed_img = heatmap * 0.4 + img_np
        return Image.fromarray(np.uint8(superimposed_img))

    def show_image_on_label(self, pil_img, label, size):
        w, h = pil_img.size
        r = min(size[0]/w, size[1]/h)
        new_size = (int(w*r), int(h*r))
        img_ctk = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=new_size)
        label.configure(image=img_ctk, text="")
        label.image = img_ctk

    def update_info_card(self, name, score):
        self.info_text.configure(state="normal")
        self.info_text.delete("0.0", "end")
        if score > 0.85:
            info = ANIMAL_INFO.get(name, {})
            text = f"KẾT QUẢ: {name.upper()}\n"
            text += f"Độ tin cậy: {score*100:.1f}%\n"
            text += f"Trạng thái: {info.get('status', '---')}\n\n"
            text += f"Mô tả: {info.get('desc', '---')}"
            self.info_text.insert("0.0", text)
        else:
            self.info_text.insert("0.0", "CẢNH BÁO: Con vật này không thuộc 9 loài động vật quý hiếm mà dự án đang theo dõi.")
        self.info_text.configure(state="disabled")

    def clear_analysis(self):
        self.cam_display.configure(image="", text="Chờ phân tích...")
        for i in range(3):
            self.prob_bars[i][0].configure(text="---")
            self.prob_bars[i][1].set(0)
        self.info_text.configure(state="normal")
        self.info_text.delete("0.0", "end")
        self.info_text.insert("0.0", "Hồ sơ động vật sẽ hiển thị ở đây...")
        self.info_text.configure(state="disabled")

    def start_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not path: return
        self.cap = cv2.VideoCapture(path)
        self.video_running = True
        self.detection_stats = {}
        self.btn_stop.configure(state="normal")
        self.btn_vid.configure(state="disabled")
        self.update_video()

    def stop_video(self):
        was_running = self.video_running
        self.video_running = False
        if self.cap: self.cap.release()
        self.btn_stop.configure(state="disabled")
        self.btn_vid.configure(state="normal")
        if was_running: self.show_final_summary()
        else: self.clear_analysis()

    def show_final_summary(self):
        if not self.detection_stats:
            self.lbl_status_bar.configure(text="KẾT QUẢ: KHÔNG XÁC ĐỊNH", text_color="#e74c3c")
            messagebox.showwarning("Thông báo", "Hệ thống không phát hiện bất kỳ loài quý hiếm nào.")
            return
        winner = max(self.detection_stats, key=self.detection_stats.get)
        self.update_info_card(winner, 1.0)
        self.lbl_status_bar.configure(text=f"KẾT QUẢ CHỐT: {winner.upper()}", text_color="#4dbd74")
        messagebox.showinfo("Kết quả", f"Hệ thống xác nhận đây là: {winner}")

    def update_video(self):
        if not self.video_running or self.cap is None: return
        ret, frame = self.cap.read()
        if not ret: 
            self.stop_video()
            return
        
        display_frame = cv2.resize(frame, (800, 450))
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb)
        
        if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) % 5 == 0:
            top3, heatmap = get_prediction_and_cam(pil_frame)
            name, score = top3[0]
            
            if score > 0.85:
                self.lbl_status_bar.configure(text=f"PHÁT HIỆN: {name} ({score*100:.0f}%)", text_color="#4dbd74")
                self.detection_stats[name] = self.detection_stats.get(name, 0) + 1
                self.update_info_card(name, score)
                
                # Live CAM
                cam_img = self.apply_heatmap(pil_frame, heatmap)
                self.show_image_on_label(cam_img, self.cam_display, (280, 180))
                
                for i, (n, p) in enumerate(top3):
                    self.prob_bars[i][0].configure(text=f"{n[:12]}")
                    self.prob_bars[i][1].set(p)
            else:
                self.lbl_status_bar.configure(text="Quét tìm động vật quý hiếm...", text_color="cyan")

        self.show_image_on_label(pil_frame, self.main_display, (800, 480))
        self.after(5, self.update_video)

if __name__ == "__main__":
    app = AnimalApp()
    app.mainloop()