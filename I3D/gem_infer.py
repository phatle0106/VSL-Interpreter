# -*- coding: utf-8 -*-

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_i3d import InceptionI3d
import torchvision.transforms as transforms
import videotransforms
import time
from collections import Counter
import os
from dotenv import load_dotenv

# =============================================================================
# CẢNH BÁO BẢO MẬT QUAN TRỌNG
# -----------------------------------------------------------------------------
# KHÔNG BAO GIỜ viết API Key trực tiếp vào code.
#
# Hướng dẫn sử dụng an toàn:
# 1. Cài đặt thư viện: pip install python-dotenv
# 2. Tạo một file mới trong cùng thư mục với code và đặt tên là ".env"
# 3. Mở file .env và thêm dòng sau (thay bằng key của bạn):
#    GEMINI_API_KEY="AIzaSy..................."
#
# Code dưới đây sẽ tự động đọc key từ file .env một cách an toàn.
# =============================================================================

# --- Tải biến môi trường từ file .env ---
load_dotenv()

# ======================= CẤU HÌNH & THAM SỐ =======================
# --- Cấu hình Mô hình & Dữ liệu ---
CLIP_LEN = 64
NUM_CLASSES = 100
WEIGHTS_PATH = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
MODE = 'rgb'
GLOSS_PATH = r'D:\Workplace\Ori_WLASL\WLASL\code\I3D\preprocess\wlasl_class_list.txt'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Load key an toàn

# --- Tham số cho Sliding Window & Xử lý hậu kỳ ---
STRIDE = 4
VOTING_BAG_SIZE = 5
THRESHOLD = 0.55
BACKGROUND_CLASS_ID = -1


# ======================= CÁC HÀM TIỆN ÍCH =======================
def load_gloss_map(path):
    gloss_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2: continue
            class_id = int(parts[0])
            gloss = ' '.join(parts[1:])
            gloss_map[class_id] = gloss
    return gloss_map

transform = transforms.Compose([videotransforms.CenterCrop(224)])

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = (frame / 255.0) * 2 - 1
    return frame

def load_model():
    print("Loading I3D model...")
    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
    model.replace_logits(NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()
    print("Model loaded successfully.")
    return model

def frames_to_tensor(frames):
    frames_np = np.stack(frames, axis=0)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)
    return frames_tensor.cuda()

# ======================= VÒNG LẶP CHÍNH =======================
def main():
    if not GEMINI_API_KEY:
        print("CẢNH BÁO: Không tìm thấy GEMINI_API_KEY. Vui lòng tạo file .env và làm theo hướng dẫn ở đầu code.")
        return
    
    gloss_map = load_gloss_map(GLOSS_PATH)
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam.")
        return

    frame_buffer = []
    raw_predictions_queue = []
    last_confirmed_class_id = None
    confirmed_gloss_text = ""
    frame_counter = 0
    
    fps_time = time.time()
    fps_count = 0
    fps = 0

    print("\nStarting real-time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ webcam.")
            break

        frame_proc = preprocess_frame(frame)
        frame_buffer.append(frame_proc)
        if len(frame_buffer) > CLIP_LEN:
            frame_buffer.pop(0)

        frame_counter += 1
        
        if len(frame_buffer) == CLIP_LEN and frame_counter % STRIDE == 0:
            with torch.no_grad():
                input_tensor = frames_to_tensor(frame_buffer)
                logits = model(input_tensor)
                predictions = torch.max(logits, dim=2)[0]
                probs = F.softmax(predictions, dim=1)
                max_prob, pred_class = torch.max(probs, dim=1)
                pred_class_id = pred_class.item()
                max_prob_val = max_prob.item()

                if max_prob_val >= THRESHOLD:
                    raw_predictions_queue.append(pred_class_id)
                else:
                    raw_predictions_queue.append(BACKGROUND_CLASS_ID)
                
                if len(raw_predictions_queue) > VOTING_BAG_SIZE:
                    raw_predictions_queue.pop(0)

            if len(raw_predictions_queue) == VOTING_BAG_SIZE:
                vote_counts = Counter(raw_predictions_queue)
                most_common_item = vote_counts.most_common(1)[0]
                majority_class_id = most_common_item[0]
                max_count = most_common_item[1]
                
                if majority_class_id != BACKGROUND_CLASS_ID and max_count > VOTING_BAG_SIZE / 2:
                    if majority_class_id != last_confirmed_class_id:
                        gloss = gloss_map.get(majority_class_id, f'Class_{majority_class_id}')
                        confirmed_gloss_text = f"{gloss}"
                        last_confirmed_class_id = majority_class_id
                        print(f"   ---> Recognized: {gloss}")
                else:
                    if last_confirmed_class_id is not None:
                         confirmed_gloss_text = ""
                    last_confirmed_class_id = None

        display_frame = frame.copy()
        
        if len(frame_buffer) < CLIP_LEN:
            status_text = "Collecting frames..."
            cv2.putText(display_frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            if confirmed_gloss_text:
                 cv2.putText(display_frame, confirmed_gloss_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        cv2.putText(display_frame, f'FPS: {fps}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Real-time Sign Language Recognition', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()