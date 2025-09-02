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
import json
from threading import Thread, Lock

# ThÃªm Mediapipe
import mediapipe as mp

# =============================================================================
# Cáº¢NH BÃO Báº¢O Máº¬T QUAN TRá»ŒNG
# =============================================================================
load_dotenv()

# ======================= Cáº¤U HÃŒNH & THAM Sá» =======================
CLIP_LEN = 64
NUM_CLASSES = 100
WEIGHTS_PATH = "checkpoint/nslt_100_005624_0.756.pt"
MODE = 'rgb'
GLOSS_PATH = r'preprocess/wlasl_class_list.txt'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API configuration
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
MIN_GLOSSES_FOR_GEMINI = 3  # Minimum glosses before sending to Gemini
SEND_TIMEOUT = 10

STRIDE = 4
VOTING_BAG_SIZE = 6
THRESHOLD = 0.6
BACKGROUND_CLASS_ID = -1

# Báº­t/táº¯t Mediapipe
USE_MEDIAPIPE = True

# Gloss collection buffer
collected_glosses = []
glosses_lock = Lock()

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
                print(f"\n   🤖 GEMINI SENTENCE: {sentence}")
                print(f"   📝 From glosses ({len(glosses_list)}): {glosses_text}\n")
                return sentence
            else:
                print("   ---> Gemini: No candidates returned")
        else:
            print(f"   ---> Gemini HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"   ---> Error contacting Gemini: {e}")
    
    return None

def add_gloss_and_check_gemini(gloss):
    """
    Add a new gloss to collection and send to Gemini if we have enough
    """
    global collected_glosses
    
    with glosses_lock:
        collected_glosses.append(gloss)
        current_count = len(collected_glosses)
        
        # If we have enough glosses, send to Gemini and clear buffer
        if current_count >= MIN_GLOSSES_FOR_GEMINI:
            glosses_to_send = list(collected_glosses)
            collected_glosses = []  # Clear buffer after copying
            
            # Send to Gemini in background thread
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
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))  # (C,T,H,W)
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)         # (1,C,T,H,W)
    return frames_tensor.cuda()

# ======================= VÃ'NG Láº¶P CHÃNH =======================
def main():
    global collected_glosses
    
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

    # Khá»Ÿi táº¡o Mediapipe náº¿u báº­t
    if USE_MEDIAPIPE:
        mp_selfie = mp.solutions.selfie_segmentation
        selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)  # chá»n model 0 nhanh hÆ¡n

    frame_buffer = []
    raw_predictions_queue = []
    last_confirmed_class_id = None
    confirmed_gloss_text = ""
    frame_counter = 0

    fps_time = time.time()
    fps_count = 0
    fps = 0

    print(f"\nStarting real-time recognition with Gemini Flash 2.5.")
    print(f"Will send to Gemini when {MIN_GLOSSES_FOR_GEMINI}+ glosses are collected.")
    print("Press 'q' to quit, 'r' to reset gloss buffer.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            with glosses_lock:
                collected_glosses = []
            print("   ---> Gloss buffer reset!")

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
                        print(f"   ✅ Recognized: {gloss}")
                        
                        # Add to collection and potentially send to Gemini
                        add_gloss_and_check_gemini(gloss)
                else:
                    confirmed_gloss_text = ""
                    last_confirmed_class_id = None

        # ======== Náº¿u báº­t Mediapipe, xá»­ lÃ½ ngáº§m ========
        if USE_MEDIAPIPE:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _ = selfie_seg.process(rgb_frame)  # cháº¡y ngáº§m, khÃ´ng dÃ¹ng mask Ä'á»ƒ hiá»ƒn thá»‹

        # ======== Overlay text ========
        display_frame = frame.copy()  # chá»‰ hiá»ƒn thá»‹ webcam bÃ¬nh thÆ°á»ng
        if len(frame_buffer) < CLIP_LEN:
            cv2.putText(display_frame, "Collecting frames...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif confirmed_gloss_text:
            cv2.putText(display_frame, confirmed_gloss_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Show current gloss buffer count
        with glosses_lock:
            current_buffer_size = len(collected_glosses)
        
        cv2.putText(display_frame, f'Glosses: {current_buffer_size} (Send at {MIN_GLOSSES_FOR_GEMINI}+)', (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show collected glosses if any
        if current_buffer_size > 0:
            with glosses_lock:
                glosses_preview = " ".join(collected_glosses[-5:])  # Show last 5 glosses
                if len(collected_glosses) > 5:
                    glosses_preview = "..." + glosses_preview
            cv2.putText(display_frame, f'Buffer: {glosses_preview}', (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        cv2.putText(display_frame, f'FPS: {fps}', (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(display_frame, "Press 'r' to reset buffer, 'q' to quit", (30, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Real-time Sign Language Recognition', display_frame)

    if USE_MEDIAPIPE:
        selfie_seg.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()