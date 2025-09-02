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
import requests
from threading import Thread, Lock
import json

# ThÃªm Mediapipe
import mediapipe as mp

# =============================================================================
# Cáº¢NH BÃO Báº¢O Máº¬T QUAN TRá»ŒNG
# =============================================================================
load_dotenv()

# Khai bÃ¡o Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
LOCAL_MIN_GLOSSES = int(os.getenv("LOCAL_MIN_GLOSSES", "3"))  # local threshold
SEND_TIMEOUT = 10

# local buffers: session_id -> list of glosses not yet sent
_local_buffers = {}
_buffers_lock = Lock()

# ======================= Cáº¤U HÃŒNH & THAM Sá» =======================
CLIP_LEN = 64
NUM_CLASSES = 100
WEIGHTS_PATH = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
MODE = 'rgb'
GLOSS_PATH = r'preprocess/wlasl_class_list.txt'

STRIDE = 4
VOTING_BAG_SIZE = 8
THRESHOLD = 0.612
BACKGROUND_CLASS_ID = -1

# Cáº¥u hÃ¬nh ná»n áº£o
USE_VIRTUAL_BG = True
BG_PATH = 'background.jpg'
dark_colors = {
    "dark_gray":   (50, 50, 50),
    "dark_blue":   (100, 50, 0),
    "dark_green":  (50, 100, 50),
    "dark_red":    (50, 50, 150),
    "dark_purple": (100, 50, 100),
    "dark_cyan":   (100, 100, 50),
    "dark_brown":  (50, 80, 120),
}
FALLBACK_BG_COLOR = dark_colors["dark_purple"]

# ======================= CÃC HÃ€M TIá»†N ÃCH =======================
def _send_gemini_request(glosses_list):
    """
    Send glosses to Gemini API and get a meaningful sentence
    """
    if not GEMINI_API_KEY:
        print("   ---> Error: GEMINI_API_KEY not found!")
        return
    
    try:
        # Create the prompt for Gemini
        glosses_text = " ".join(glosses_list)
        prompt = f"""You are a sign language interpreter. I will give you a sequence of sign language glosses (individual sign words), and you need to convert them into a natural, grammatically correct English sentence that conveys the intended meaning.

Glosses: {glosses_text}

Please provide a natural English sentence that represents what the person is trying to communicate through these signs. Focus on the meaning rather than literal word order, as sign language grammar differs from English grammar.

Respond with only the sentence, no additional explanation."""

        # Prepare the request payload for Gemini API
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 100,
            }
        }

        # Make the API request
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(GEMINI_API_URL, 
                               json=payload, 
                               headers=headers, 
                               timeout=SEND_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                sentence = result['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f"   ---> Gemini sentence: {sentence}")
                print(f"   ---> From glosses: {glosses_text}")
                return sentence
            else:
                print("   ---> Gemini: No candidates returned")
        else:
            print(f"   ---> Gemini HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"   ---> Error contacting Gemini: {e}")
    
    return None

def enqueue_gloss_and_maybe_send(gloss, session_id="default", reset=False):
    """
    Append to local buffer for session; if buffer length >= LOCAL_MIN_GLOSSES,
    send payload (in background) and then clear local buffer for that session.
    """
    gloss = str(gloss).strip()
    if gloss == "":
        return

    with _buffers_lock:
        if reset or session_id not in _local_buffers:
            _local_buffers[session_id] = []
        _local_buffers[session_id].append(gloss)
        current_len = len(_local_buffers[session_id])

        # If we've reached threshold, prepare payload and clear local buffer
        if current_len >= LOCAL_MIN_GLOSSES:
            glosses_to_send = list(_local_buffers[session_id])
            _local_buffers[session_id] = []  # clear after taking
        else:
            glosses_to_send = None

    if glosses_to_send:
        # Send non-blocking so main loop isn't blocked
        Thread(target=_send_gemini_request, args=(glosses_to_send,), daemon=True).start()

def load_gloss_map(path):
    gloss_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
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
    frames_np = np.stack(frames, axis=0)                # (T,H,W,C)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))   # (C,T,H,W)
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)          # (1,C,T,H,W)
    return frames_tensor.cuda()

# ======================= VÃ'NG Láº¶P CHÃNH =======================
def main():
    if not GEMINI_API_KEY:
        print("Cáº¢NH BÃO: KhÃ´ng tÃ¬m tháº¥y GEMINI_API_KEY. Vui lÃ²ng táº¡o file .env vÃ  thÃªm GEMINI_API_KEY=your_api_key_here")
        print("Báº¡n cÃ³ thá»ƒ láº¥y API key táº¡i: https://makersuite.google.com/app/apikey")
        return

    gloss_map = load_gloss_map(GLOSS_PATH)
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lá»—i: KhÃ´ng thá»ƒ má»Ÿ webcam.")
        return

    # âœ… Mediapipe segmentation (khá»Ÿi táº¡o 1 láº§n)
    mp_selfie = mp.solutions.selfie_segmentation
    selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)

    # Chuáº©n bá»‹ áº£nh ná»n
    bg_image = None
    if USE_VIRTUAL_BG and os.path.exists(BG_PATH):
        bg_image = cv2.imread(BG_PATH)
    if bg_image is None:
        bg_resized_static = None
    else:
        bg_resized_static = bg_image

    frame_buffer = []
    raw_predictions_queue = []
    last_confirmed_class_id = None
    confirmed_gloss_text = ""
    frame_counter = 0

    fps_time = time.time()
    fps_count = 0
    fps = 0

    print(f"\nStarting real-time recognition with Gemini Flash 2.5.")
    print(f"Will send glosses to Gemini after collecting {LOCAL_MIN_GLOSSES} signs.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # resize background fallback
        if bg_resized_static is not None:
            bg_resized = cv2.resize(bg_resized_static, (W, H))
        else:
            bg_resized = np.full((H, W, 3), FALLBACK_BG_COLOR, dtype=np.uint8)

        # ======== Inference (I3D) ========
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
                majority_class_id, max_count = vote_counts.most_common(1)[0]
                if majority_class_id != BACKGROUND_CLASS_ID and max_count > VOTING_BAG_SIZE / 2:
                    if majority_class_id != last_confirmed_class_id:
                        gloss = gloss_map.get(majority_class_id, f'Class_{majority_class_id}')
                        confirmed_gloss_text = gloss
                        last_confirmed_class_id = majority_class_id
                        print(f"   ---> Recognized: {gloss}")

                        # Enqueue; will send to Gemini when we have >= LOCAL_MIN_GLOSSES
                        enqueue_gloss_and_maybe_send(gloss, session_id="default")
                else:
                    confirmed_gloss_text = ""
                    last_confirmed_class_id = None

        # ======== Mediapipe background replace ========
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_seg.process(rgb_frame)
        mask = results.segmentation_mask  # float32 [0..1]

        # LÃ m mÆ°á»£t mask
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        # âœ… Threshold tháº¥p hÆ¡n Ä'á»ƒ mask rá»™ng hÆ¡n
        mask = (mask > 0.40).astype(np.uint8)

        # âœ… Ná»›i rá»™ng mask thÃªm báº±ng dilate
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Apply mask
        mask_3c = mask[..., None]
        display_frame = np.where(mask_3c == 1, frame, bg_resized)

        # ======== Overlay text ========
        if len(frame_buffer) < CLIP_LEN:
            cv2.putText(display_frame, "Collecting frames...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif confirmed_gloss_text:
            cv2.putText(display_frame, confirmed_gloss_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Show current buffer count
        with _buffers_lock:
            current_buffer_size = len(_local_buffers.get("default", []))
        
        cv2.putText(display_frame, f'Glosses: {current_buffer_size}/{LOCAL_MIN_GLOSSES}', (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        cv2.putText(display_frame, f'FPS: {fps}', (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Real-time Sign Language Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    selfie_seg.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()