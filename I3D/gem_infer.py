# -*- coding: utf-8 -*-

import streamlit as st
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
from PIL import Image
import base64
from io import BytesIO

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================
load_dotenv()

# Model parameters
CLIP_LEN = 64
NUM_CLASSES = 100
WEIGHTS_PATH = "checkpoint/nslt_100_005624_0.756.pt"
MODE = 'rgb'
GLOSS_PATH = r'preprocess/wlasl_class_list.txt'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API configuration
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
SEND_TIMEOUT = 10

# Recognition parameters
STRIDE = 5
VOTING_BAG_SIZE = 6
THRESHOLD = 0.605
BACKGROUND_CLASS_ID = -1

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ASL Recognition with AI Sentence Generation",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching the HTML design
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .video-container {
        background: linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.01)), #0b1220;
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 24px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,.35);
    }
    
    .video-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(148,163,184,.18);
        background: rgba(255,255,255,.02);
    }
    
    .video-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        color: #e5e7eb;
    }
    
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #ef4444;
        display: inline-block;
    }
    
    .status-dot.connected {
        background: #22c55e;
        box-shadow: 0 0 10px rgba(34,197,94,.3);
    }
    
    .badge {
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,.18);
        color: #8ea0b8;
        background: rgba(255,255,255,.03);
        margin: 0 4px;
    }
    
    .badge.ok {
        color: #16a34a;
        border-color: rgba(22,163,74,.45);
    }
    
    .control-panel {
        background: linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.01)), #0b1220;
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 24px;
        padding: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,.35);
    }
    
    .info-panel {
        background: rgba(255,255,255,.03);
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .info-title {
        font-size: 12px;
        color: #8ea0b8;
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .info-content {
        font-size: 16px;
        color: #e5e7eb;
        font-family: 'Courier New', monospace;
        min-height: 24px;
        word-break: break-word;
    }
    
    .sentence-display {
        background: linear-gradient(135deg, rgba(34,197,94,.15), rgba(59,130,246,.1));
        border: 2px solid rgba(34,197,94,.3);
    }
    
    .sentence-display .info-content {
        font-weight: 600;
        font-size: 18px;
        color: #22c55e;
    }
    
    .messages-container {
        background: rgba(255,255,255,.02);
        border-radius: 12px;
        padding: 16px;
        max-height: 400px;
        overflow-y: auto;
        margin-top: 16px;
    }
    
    .message {
        margin: 8px 0;
        padding: 12px 16px;
        border-radius: 12px;
        border: 1px solid rgba(148,163,184,.18);
        max-width: 85%;
    }
    
    .message.gloss {
        background: rgba(168,85,247,.12);
        border-color: rgba(168,85,247,.35);
        margin-left: auto;
        font-family: 'Courier New', monospace;
        text-align: right;
        color: #a855f7;
    }
    
    .message.sentence {
        background: rgba(34,197,94,.15);
        border-color: rgba(34,197,94,.4);
        margin: 0 auto;
        font-weight: 600;
        color: #22c55e;
        text-align: center;
    }
    
    .message.system {
        background: rgba(59,130,246,.12);
        border-color: rgba(59,130,246,.35);
        margin: 0 auto;
        color: #3b82f6;
        font-size: 14px;
        text-align: center;
    }
    
    .stats-overlay {
        position: absolute;
        top: 8px;
        left: 8px;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-family: monospace;
    }
    
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #22c55e, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(34,197,94,.3);
    }
    
    .secondary-btn > button {
        background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    }
    
    .danger-btn > button {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'glosses_buffer' not in st.session_state:
    st.session_state.glosses_buffer = []

if 'model' not in st.session_state:
    st.session_state.model = None

if 'gloss_map' not in st.session_state:
    st.session_state.gloss_map = {}

if 'frame_buffer' not in st.session_state:
    st.session_state.frame_buffer = []

if 'raw_predictions_queue' not in st.session_state:
    st.session_state.raw_predictions_queue = []

if 'last_confirmed_class_id' not in st.session_state:
    st.session_state.last_confirmed_class_id = None

if 'confirmed_gloss_text' not in st.session_state:
    st.session_state.confirmed_gloss_text = ""

if 'frame_counter' not in st.session_state:
    st.session_state.frame_counter = 0

if 'generated_sentences' not in st.session_state:
    st.session_state.generated_sentences = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

if 'ai_connected' not in st.session_state:
    st.session_state.ai_connected = False

if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = "Ready to generate..."

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
@st.experimental_singleton
def load_gloss_map(path):
    """Load gloss mapping from file"""
    gloss_map = {}
    try:
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
    except FileNotFoundError:
        st.error(f"Gloss file not found: {path}")
        st.stop()
    return gloss_map

@st.experimental_singleton
def load_model():
    """Load the I3D model"""
    try:
        with st.spinner("Loading I3D model..."):
            model = InceptionI3d(400, in_channels=3)
            model.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location='cpu'))
            model.replace_logits(NUM_CLASSES)
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
            if torch.cuda.is_available():
                model.cuda()
            model = torch.nn.DataParallel(model)
            model.eval()
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    frame = cv2.resize(frame, (224, 224))
    frame = (frame / 255.0) * 2 - 1
    return frame

def frames_to_tensor(frames):
    """Convert frames list to tensor"""
    transform = transforms.Compose([videotransforms.CenterCrop(224)])
    frames_np = np.stack(frames, axis=0)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)
    
    if torch.cuda.is_available():
        return frames_tensor.cuda()
    return frames_tensor

def send_gemini_request(glosses_list):
    """Send glosses to Gemini API and get a meaningful sentence"""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not found!"
    
    try:
        glosses_text = " ".join(glosses_list)
        prompt = f"""You are a sign language interpreter. I will give you a sequence of sign language glosses (individual sign words), and you need to convert them into a natural, grammatically correct English sentence that conveys the intended meaning.

Glosses: {glosses_text}

Please provide a natural English sentence that represents what the person is trying to communicate through these signs. Focus on the meaning rather than literal word order, as sign language grammar differs from English grammar.

Respond with only the sentence, no additional explanation."""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 100,
            }
        }

        headers = {"Content-Type": "application/json"}
        
        response = requests.post(GEMINI_API_URL, 
                               json=payload, 
                               headers=headers, 
                               timeout=SEND_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                sentence = result['candidates'][0]['content']['parts'][0]['text'].strip()
                return sentence
            else:
                return "Gemini: No response generated"
        else:
            return f"Gemini HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def add_message(text, msg_type='system'):
    """Add message to session state"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.messages.append({
        'text': text,
        'type': msg_type,
        'timestamp': timestamp
    })

def process_frame_for_recognition(frame):
    """Process frame and return recognition result"""
    if st.session_state.model is None:
        return None
        
    frame_proc = preprocess_frame(frame)
    st.session_state.frame_buffer.append(frame_proc)
    
    if len(st.session_state.frame_buffer) > CLIP_LEN:
        st.session_state.frame_buffer.pop(0)
    
    st.session_state.frame_counter += 1
    
    if len(st.session_state.frame_buffer) == CLIP_LEN and st.session_state.frame_counter % STRIDE == 0:
        with torch.no_grad():
            input_tensor = frames_to_tensor(st.session_state.frame_buffer)
            logits = st.session_state.model(input_tensor)
            predictions = torch.max(logits, dim=2)[0]
            probs = F.softmax(predictions, dim=1)
            max_prob, pred_class = torch.max(probs, dim=1)
            pred_class_id = pred_class.item()
            max_prob_val = max_prob.item()

            if max_prob_val >= THRESHOLD:
                st.session_state.raw_predictions_queue.append(pred_class_id)
            else:
                st.session_state.raw_predictions_queue.append(BACKGROUND_CLASS_ID)

            if len(st.session_state.raw_predictions_queue) > VOTING_BAG_SIZE:
                st.session_state.raw_predictions_queue.pop(0)

        if len(st.session_state.raw_predictions_queue) == VOTING_BAG_SIZE:
            vote_counts = Counter(st.session_state.raw_predictions_queue)
            majority_class_id, max_count = vote_counts.most_common(1)[0]
            
            if majority_class_id != BACKGROUND_CLASS_ID and max_count > VOTING_BAG_SIZE / 2:
                if majority_class_id != st.session_state.last_confirmed_class_id:
                    gloss = st.session_state.gloss_map.get(majority_class_id, f'Class_{majority_class_id}')
                    st.session_state.confirmed_gloss_text = gloss
                    st.session_state.last_confirmed_class_id = majority_class_id
                    
                    st.session_state.glosses_buffer.append(gloss)
                    add_message(f"Recognized: {gloss}", 'gloss')
                    return gloss
            else:
                st.session_state.confirmed_gloss_text = ""
                st.session_state.last_confirmed_class_id = None
    
    return None

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div class="video-header" style="margin-bottom: 20px;">
        <div class="video-title">
            <span class="status-dot connected"></span>
            <span>ASL Recognition with AI Sentence Generation</span>
        </div>
        <div>
            <span class="badge ok">I3D Ready</span>
            <span class="badge ok">Gemini Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API key
    if not GEMINI_API_KEY:
        st.error("⚠️ GEMINI_API_KEY not found! Please create a .env file with your API key.")
        st.markdown("Get your API key at: https://makersuite.google.com/app/apikey")
        return
    
    # Load model and gloss map
    if st.session_state.model is None:
        st.session_state.model = load_model()
        st.session_state.gloss_map = load_gloss_map(GLOSS_PATH)
        if st.session_state.model is not None:
            add_message("AI models loaded successfully!", 'system')
# Layout 2 cột chính
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        # Video container
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        camera_input = st.camera_input("ASL Recognition Camera", key="asl_camera")

        if camera_input is not None:
            image = Image.open(camera_input)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if st.session_state.model is not None:
                recognized_gloss = process_frame_for_recognition(frame)

            if len(st.session_state.frame_buffer) < CLIP_LEN:
                st.info(f"Collecting frames... ({len(st.session_state.frame_buffer)}/{CLIP_LEN})")
            else:
                st.success("Ready for recognition")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-panel">
            <div class="info-title">System Status</div>
            <div class="info-content">
                Glosses Collected: {len(st.session_state.glosses_buffer)} | 
                Sentences Generated: {len(st.session_state.generated_sentences)} | 
                Frame Buffer: {len(st.session_state.frame_buffer)}/{CLIP_LEN}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 👉 Hàng nút tách riêng, KHÔNG đặt trong col2
    spacer1, spacer2, col_btn1, col_btn2, col_btn3 = st.columns([1.1, 0.9, 1, 1, 1])

    with col_btn1:
        if st.button("🤖 Generate Sentence", disabled=len(st.session_state.glosses_buffer) == 0):
            if st.session_state.glosses_buffer:
                with st.spinner("Generating sentence..."):
                    sentence = send_gemini_request(st.session_state.glosses_buffer)
                    st.session_state.current_sentence = sentence
                    st.session_state.generated_sentences.append({
                        "glosses": list(st.session_state.glosses_buffer),
                        "sentence": sentence,
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                    add_message(sentence, 'sentence')
                    st.session_state.glosses_buffer = []
                st.success("Sentence generated!")

    with col_btn2:
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Clear Glosses"):
            st.session_state.glosses_buffer = []
            add_message("Glosses cleared", 'system')
            st.success("Glosses cleared!")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_btn3:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("🔄 Reset All"):
            st.session_state.glosses_buffer = []
            st.session_state.messages = []
            st.session_state.generated_sentences = []
            st.session_state.current_sentence = "Ready to generate..."
            st.session_state.frame_buffer = []
            st.success("Session reset!")
        st.markdown('</div>', unsafe_allow_html=True)

    # Panel bên phải
    with col2:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-panel">
            <div class="info-title">Current Glosses ({len(st.session_state.glosses_buffer)})</div>
            <div class="info-content">{' '.join(st.session_state.glosses_buffer) if st.session_state.glosses_buffer else 'No glosses collected yet'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-panel sentence-display">
            <div class="info-title">Latest Generated Sentence</div>
            <div class="info-content">{st.session_state.current_sentence}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="messages-container">', unsafe_allow_html=True)
        st.markdown("**Activity Log:**")

        for msg in st.session_state.messages[-10:]:
            msg_class = f"message {msg['type']}"
            if msg['type'] == 'gloss':
                content = f"**Gloss:** {msg['text']}"
            elif msg['type'] == 'sentence':
                content = f"**Generated:** \"{msg['text']}\""
            else:
                content = msg['text']

            st.markdown(f"""
            <div class="{msg_class}">
                <small>{msg['timestamp']}</small><br>
                {content}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
    # Instructions
    with st.expander("📋 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. **Allow camera access** when prompted by your browser
        2. **Position yourself** clearly in front of the camera
        3. **Perform sign language** - glosses will be recognized automatically
        4. **Generate sentences** by clicking the button when you have enough glosses
        
        **Tips:**
        - Ensure good lighting for better recognition
        - Wait for the frame buffer to fill (64 frames)
        - Clear glosses to start a new sequence
        - Use Reset All to clear everything
        
        **Recognition Process:**
        - The system uses I3D neural network for real-time ASL recognition
        - Glosses are collected automatically as you sign
        - Gemini AI converts gloss sequences into natural sentences
        """)

if __name__ == '__main__':
    main()