import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageFile
import cv2
import tempfile
import os
import numpy as np

# Cho phep load anh bi loi stream (sua loi broken data stream truoc do)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. CAU HINH HE THONG
st.set_page_config(page_title="AI Animal Recognition", layout="wide")
st.title("HE THONG NHAN DIEN DONG VAT NGUY CAP")
st.write("Cong nghe: ResNet-18 Transfer Learning | Doi tuong: 9 loai dong vat hoang da")

# Danh sach class (phai trung thu tu voi luc huan luyen)
CLASS_NAMES = [
    'Bao_Gam', 'Cu_Li_Nho', 'Gau_Ngua', 'Ho_Dong_Duong', 
    'Huou_Sao', 'Te_Te_Java', 'Voi_Chau_A', 'Vooc_Cha_Va', 'Vuon_Ma_Vang'
]

# 2. TAI MO HINH
@st.cache_resource
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        model.to(device)
        return model, device
    else:
        st.error("Khong tim thay file ResNet18_Best_Weights.pth. Vui long kiem tra lai thu muc.")
        return None, device

model_data = load_model()

# 3. TIEN XU LY ANH
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image, model, device):
    # 1. Ảnh gốc
    img_tensor1 = data_transform(image).unsqueeze(0).to(device)
    
    # 2. Ảnh lật ngang (TTA)
    flipped_img = image.transpose(Image.FLIP_LEFT_RIGHT)
    img_tensor2 = data_transform(flipped_img).unsqueeze(0).to(device)
    
    # 3. Ảnh làm mịn song phương (Bilateral Filter - loại bỏ lồng lưới sắt)
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        blurred = cv2.bilateralFilter(bgr, 9, 75, 75)
        rgb_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        pil_img_blurred = Image.fromarray(rgb_blurred)
    else:
        pil_img_blurred = image
    img_tensor3 = data_transform(pil_img_blurred).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out1 = model(img_tensor1)
        probs1 = torch.nn.functional.softmax(out1, dim=1)[0]
        
        out2 = model(img_tensor2)
        probs2 = torch.nn.functional.softmax(out2, dim=1)[0]
        
        out3 = model(img_tensor3)
        probs3 = torch.nn.functional.softmax(out3, dim=1)[0]
        
        # Cộng trung bình xác suất TTA
        probs = (probs1 + probs2 + probs3) / 3.0
        confidence, preds = torch.max(probs.unsqueeze(0), 1)
        
    return CLASS_NAMES[preds[0].item()], confidence.item()


# 4. GIAO DIEN TUONG TAC
if model_data[0] is not None:
    model, device = model_data
    
    # Khu vuc tai file
    st.sidebar.header("Bang dieu khien")
    uploaded_file = st.sidebar.file_uploader("Tai anh hoac video len", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
    min_confidence = st.sidebar.slider("Nguong tin cay (%)", 30, 100, 70) / 100.0

    if uploaded_file is not None:
        # XU LY HINH ANH
        if uploaded_file.type.startswith('image'):
            col1, col2 = st.columns([1, 1])
            image = Image.open(uploaded_file).convert('RGB')
            
            with col1:
                st.image(image, caption="Anh dau vao", use_column_width=True)
            
            with col2:
                st.subheader("Ket qua phan tich")
                label, score = predict_image(image, model, device)
                if score >= min_confidence:
                    st.success(f"Doi tuong: {label}")
                    st.write(f"Do tin cay: {score*100:.2f}%")
                    st.progress(score)
                else:
                    st.warning("Khong tim thay doi tuong nao du do tin cay.")

        # XU LY VIDEO
        elif uploaded_file.type.startswith('video'):
            st.subheader("Phan tich video theo thoi gian thuc")
            
            # Luu tam file video de OpenCV co the doc
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.info(f"Video FPS: {fps} | Tong so khung hinh: {total_frames}")
            
            # Tao khu vuc hien thi timeline
            timeline_placeholder = st.empty()
            gallery_container = st.container()
            gallery_container.write("Cac thoi diem phat hien dong vat:")
            
            cols = gallery_container.columns(4)
            found_count = 0
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Chi phan tich 1 khung hinh moi giay de tang toc do
                if frame_idx % fps == 0:
                    current_sec = frame_idx // fps
                    
                    # Chuyen doi sang RGB va PIL
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    label, score = predict_image(pil_img, model, device)
                    
                    if score >= min_confidence:
                        with cols[found_count % 4]:
                            st.image(pil_img, caption=f"{current_sec}s: {label} ({score*100:.0f}%)", use_column_width=True)
                        found_count += 1
                
                frame_idx += 1
            
            cap.release()
            os.remove(tfile.name) # Xoa file tam
            st.write(f"Tong cong tim thay {found_count} khoanh khac co dong vat.")
    else:
        st.info("Vui long tai len file anh hoac video tu thanh ben trai de bat dau.")