# -*- coding: utf-8 -*-

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_i3d import InceptionI3d
import torchvision.transforms as transforms
import videotransforms
import time
from collections import Counter, deque
import os
from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, T5Tokenizer
import threading
import queue

# =============================================================================
# CẢNH BÁO BẢO MẬT QUAN TRỌNG
# -----------------------------------------------------------------------------
# KHÔNG BAO GIỜ viết API Key trực tiếp vào code.
#
# Hướng dẫn sử dụng an toàn:
# 1. Cài đặt thư viện: pip install python-dotenv transformers
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

# --- Tham số cho tính năng tạo câu ---
T5_MODEL_NAME = "t5-small"  # Có thể thay đổi thành "t5-base" hoặc "t5-large" để có chất lượng tốt hơn
WORD_COLLECTION_TIME = 10.0  # Thời gian thu thập từ (giây) trước khi tạo câu
MIN_WORDS_FOR_SENTENCE = 3   # Số từ tối thiểu để tạo câu
MAX_WORDS_FOR_SENTENCE = 10  # Số từ tối đa để tạo câu
SENTENCE_GENERATION_ENABLED = True

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

def load_t5_model():
    """Load T5 model for sentence generation"""
    if not SENTENCE_GENERATION_ENABLED:
        return None, None
    
    print("Loading T5 model for sentence generation...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        print("T5 model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading T5 model: {e}")
        print("Sentence generation will be disabled.")
        return None, None

def frames_to_tensor(frames):
    frames_np = np.stack(frames, axis=0)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)
    return frames_tensor.cuda()

class SentenceGenerator:
    """Class to handle sentence generation from collected words"""
    
    def __init__(self, t5_model, t5_tokenizer):
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.word_queue = deque(maxlen=MAX_WORDS_FOR_SENTENCE)
        self.last_sentence_time = time.time()
        self.current_sentence = ""
        self.sentence_history = deque(maxlen=5)  # Keep last 5 sentences
        
    def add_word(self, word):
        """Add a recognized word to the collection"""
        current_time = time.time()
        
        # Add word with timestamp
        self.word_queue.append((word, current_time))
        
        # Check if it's time to generate a sentence
        if (current_time - self.last_sentence_time > WORD_COLLECTION_TIME and 
            len(self.word_queue) >= MIN_WORDS_FOR_SENTENCE):
            self.generate_sentence()
    
    def generate_sentence(self):
        """Generate a meaningful sentence from collected words"""
        if not self.t5_model or not self.t5_tokenizer or len(self.word_queue) < MIN_WORDS_FOR_SENTENCE:
            return
        
        try:
            # Extract words from queue
            words = [word for word, timestamp in self.word_queue]
            
            # Create input text for T5
            words_text = " ".join(words)
            input_text = f"generate sentence from words: {words_text}"
            
            # Tokenize input
            input_ids = self.t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            # Generate sentence
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    input_ids,
                    max_length=100,
                    min_length=10,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.t5_tokenizer.eos_token_id
                )
            
            # Decode generated sentence
            generated_sentence = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and format the sentence
            generated_sentence = generated_sentence.strip()
            if generated_sentence and not generated_sentence.endswith('.'):
                generated_sentence += '.'
            
            # Update current sentence and history
            self.current_sentence = generated_sentence
            self.sentence_history.append(f"Words: {words_text} → {generated_sentence}")
            
            # Clear word queue and reset timer
            self.word_queue.clear()
            self.last_sentence_time = time.time()
            
            print(f"   ---> Generated Sentence: {generated_sentence}")
            
        except Exception as e:
            print(f"Error generating sentence: {e}")
    
    def get_current_sentence(self):
        """Get the current generated sentence"""
        return self.current_sentence
    
    def get_sentence_history(self):
        """Get history of generated sentences"""
        return list(self.sentence_history)
    
    def reset(self):
        """Reset the sentence generator"""
        self.word_queue.clear()
        self.current_sentence = ""
        self.last_sentence_time = time.time()

# ======================= VÒNG LẶP CHÍNH =======================
def main():
    if not GEMINI_API_KEY:
        print("CẢNH BÁO: Không tìm thấy GEMINI_API_KEY. Vui lòng tạo file .env và làm theo hướng dẫn ở đầu code.")
        return
    
    gloss_map = load_gloss_map(GLOSS_PATH)
    model = load_model()
    
    # Load T5 model for sentence generation
    t5_model, t5_tokenizer = load_t5_model()
    sentence_generator = SentenceGenerator(t5_model, t5_tokenizer) if t5_model else None

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
    
    # For displaying sentence history
    show_history = False

    print("\nStarting real-time recognition. Press 'q' to quit, 'h' to toggle sentence history, 'r' to reset sentence generator.")

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
                        
                        # Add word to sentence generator
                        if sentence_generator:
                            sentence_generator.add_word(gloss)
                else:
                    if last_confirmed_class_id is not None:
                         confirmed_gloss_text = ""
                    last_confirmed_class_id = None

        # Create display frame
        display_frame = frame.copy()
        
        # Status and word recognition display
        if len(frame_buffer) < CLIP_LEN:
            status_text = "Collecting frames..."
            cv2.putText(display_frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            if confirmed_gloss_text:
                cv2.putText(display_frame, f"Word: {confirmed_gloss_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Display generated sentence
        if sentence_generator and sentence_generator.get_current_sentence():
            sentence = sentence_generator.get_current_sentence()
            # Wrap long sentences
            if len(sentence) > 60:
                mid_point = len(sentence) // 2
                space_idx = sentence.find(' ', mid_point)
                if space_idx != -1:
                    cv2.putText(display_frame, f"Sentence: {sentence[:space_idx]}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    cv2.putText(display_frame, sentence[space_idx+1:], (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                else:
                    cv2.putText(display_frame, f"Sentence: {sentence}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            else:
                cv2.putText(display_frame, f"Sentence: {sentence}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Display sentence history if toggled
        if show_history and sentence_generator:
            history = sentence_generator.get_sentence_history()
            y_pos = 150
            cv2.putText(display_frame, "--- Sentence History ---", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for i, hist_sentence in enumerate(history[-3:]):  # Show last 3
                y_pos += 30
                if len(hist_sentence) > 80:
                    hist_sentence = hist_sentence[:77] + "..."
                cv2.putText(display_frame, hist_sentence, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # FPS counter
        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        
        cv2.putText(display_frame, f'FPS: {fps}', (display_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Controls info
        cv2.putText(display_frame, "Controls: q=quit, h=history, r=reset", (30, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Real-time Sign Language Recognition with Sentence Generation', display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_history = not show_history
            print(f"Sentence history display: {'ON' if show_history else 'OFF'}")
        elif key == ord('r'):
            if sentence_generator:
                sentence_generator.reset()
                print("Sentence generator reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()