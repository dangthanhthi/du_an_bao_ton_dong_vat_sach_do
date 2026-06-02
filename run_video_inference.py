import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk, ImageFile
import os

# Ho tro doc anh bi loi cau truc
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("[*] Dang khoi dong Phan mem Nhan dien...")

# --- 1. CAU HINH MO HINH AI ---
CLASS_NAMES = [
    'Bao_Gam', 'Cu_Li_Nho', 'Gau_Ngua', 'Ho_Dong_Duong', 
    'Huou_Sao', 'Te_Te_Java', 'Voi_Chau_A', 'Vooc_Cha_Va', 'Vuon_Ma_Vang'
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, len(CLASS_NAMES))
    )
    
    weights_path = 'ResNet18_Best_Weights.pth'
    if not os.path.exists(weights_path):
        return None
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(pil_img):
    img_tensor = data_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, 1)
    return CLASS_NAMES[preds[0]], confidence.item()

# --- 2. THIET KE GIAO DIEN PHAN MEM ---
class AnimalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("He Thong Nhan Dien Dong Vat Hoang Da")
        self.root.geometry("800x650")
        
        # Kiem tra xem co file trong so chua
        if model is None:
            messagebox.showerror("Loi nghiem trong", "Khong tim thay file ResNet18_Best_Weights.pth!\nVui long de file vao cung thu muc voi code.")
            self.root.destroy()
            return

        # Tieu de phan mem
        self.lbl_title = tk.Label(root, text="PHAN MEM NHAN DIEN DONG VAT NGUY CAP", font=("Arial", 16, "bold"))
        self.lbl_title.pack(pady=20)
        
        # Khu vuc chua nut bam
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)
        
        self.btn_img = tk.Button(self.btn_frame, text="Chon Anh (*.jpg)", font=("Arial", 12), command=self.process_image, width=15, bg="lightblue")
        self.btn_img.pack(side=tk.LEFT, padx=20)
        
        self.btn_vid = tk.Button(self.btn_frame, text="Chon Video (*.mp4)", font=("Arial", 12), command=self.process_video, width=15, bg="lightgreen")
        self.btn_vid.pack(side=tk.LEFT, padx=20)
        
        # Hien thi ket qua bang chu
        self.lbl_result = tk.Label(root, text="Hay chon mot tam anh hoac video de bat dau", font=("Arial", 14), fg="red")
        self.lbl_result.pack(pady=20)
        
        # Khu vuc hien thi anh
        self.lbl_image = tk.Label(root)
        self.lbl_image.pack(pady=10)

    def process_image(self):
        # Mo cua so chon file
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
            
        self.lbl_result.config(text="AI dang phan tich anh...", fg="black")
        self.root.update()
        
        # Xu ly va du doan
        img = Image.open(file_path).convert('RGB')
        label, score = predict_image(img)
        
        # Hien thi ket qua
        if score > 0.5:
            self.lbl_result.config(text=f"Phat hien: {label} (Do tin cay: {score*100:.2f}%)", fg="green")
        else:
            self.lbl_result.config(text="Khong nhan dien duoc dong vat nao ro rang.", fg="red")
        
        # Resize anh de dua len giao dien
        img.thumbnail((450, 450))
        img_tk = ImageTk.PhotoImage(img)
        self.lbl_image.config(image=img_tk)
        self.lbl_image.image = img_tk # Luu bien de anh khong bi bien mat

    def process_video(self):
        # Mo cua so chon file video
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not file_path:
            return
            
        self.lbl_result.config(text="Dang phat video... Nhan phim 'q' de thoat video.", fg="blue")
        self.root.update()
        
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Phat video va nhan dien tung khung hinh
            display_frame = cv2.resize(frame, (960, 540))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            label, score = predict_image(pil_img)
            
            if score > 0.65:
                text = f"{label}: {score*100:.1f}%"
                cv2.rectangle(display_frame, (15, 15), (450, 70), (0, 0, 0), -1)
                cv2.putText(display_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
            cv2.imshow("Cua so Nhan dien Video (An 'q' de thoat)", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.lbl_result.config(text="Da dong video.", fg="black")

if __name__ == "__main__":
    # Khoi tao ung dung
    root = tk.Tk()
    app = AnimalApp(root)
    root.mainloop()