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
from threading import Thread, Lock
import mediapipe as mp
from PIL import Image

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

# Mediapipe settings
USE_MEDIAPIPE = True

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
@st.cache_resource
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

@st.cache_resource
def load_model():
    """Load the I3D model"""
    try:
        with st.spinner("Loading I3D model..."):
            model = InceptionI3d(400, in_channels=3)
            model.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
            model.replace_logits(NUM_CLASSES)
            model.load_state_dict(torch.load(WEIGHTS_PATH))
            model.cuda() if torch.cuda.is_available() else model.cpu()
            model = torch.nn.DataParallel(model)
            model.eval()
            st.success("Model loaded successfully!")
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource
def initialize_mediapipe():
    """Initialize MediaPipe selfie segmentation"""
    if USE_MEDIAPIPE:
        mp_selfie = mp.solutions.selfie_segmentation
        return mp_selfie.SelfieSegmentation(model_selection=1)
    return None

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    frame = cv2.resize(frame, (224, 224))
    frame = (frame / 255.0) * 2 - 1
    return frame

def frames_to_tensor(frames):
    """Convert frames list to tensor"""
    transform = transforms.Compose([videotransforms.CenterCrop(224)])
    frames_np = np.stack(frames, axis=0)                # (T,H,W,C)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))  # (C,T,H,W)
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)         # (1,C,T,H,W)
    
    if torch.cuda.is_available():
        return frames_tensor.cuda()
    return frames_tensor

def send_gemini_request(glosses_list):
    """Send glosses to Gemini API and get a meaningful sentence"""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not found!"
    
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
                return sentence
            else:
                return "Gemini: No candidates returned"
        else:
            return f"Gemini HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Error contacting Gemini: {e}"

def process_frame_for_recognition(frame):
    """Process frame and return recognition result"""
    # Preprocess frame
    frame_proc = preprocess_frame(frame)
    st.session_state.frame_buffer.append(frame_proc)
    
    if len(st.session_state.frame_buffer) > CLIP_LEN:
        st.session_state.frame_buffer.pop(0)
    
    st.session_state.frame_counter += 1
    
    # Perform inference if we have enough frames
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

        # Voting mechanism
        if len(st.session_state.raw_predictions_queue) == VOTING_BAG_SIZE:
            vote_counts = Counter(st.session_state.raw_predictions_queue)
            majority_class_id, max_count = vote_counts.most_common(1)[0]
            
            if majority_class_id != BACKGROUND_CLASS_ID and max_count > VOTING_BAG_SIZE / 2:
                if majority_class_id != st.session_state.last_confirmed_class_id:
                    gloss = st.session_state.gloss_map.get(majority_class_id, f'Class_{majority_class_id}')
                    st.session_state.confirmed_gloss_text = gloss
                    st.session_state.last_confirmed_class_id = majority_class_id
                    
                    # Add to glosses buffer
                    st.session_state.glosses_buffer.append(gloss)
                    return gloss
            else:
                st.session_state.confirmed_gloss_text = ""
                st.session_state.last_confirmed_class_id = None
    
    return None

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.title("🤟 Real-time Sign Language Recognition")
    st.markdown("---")
    
    # Check API key
    if not GEMINI_API_KEY:
        st.error("⚠️ GEMINI_API_KEY not found! Please create a .env file with GEMINI_API_KEY=your_api_key_here")
        st.markdown("Get your API key at: https://makersuite.google.com/app/apikey")
        st.stop()
    
    # Load model and gloss map
    if st.session_state.model is None:
        st.session_state.model = load_model()
        st.session_state.gloss_map = load_gloss_map(GLOSS_PATH)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Camera controls
        st.subheader("📹 Camera")
        camera_enabled = st.checkbox("Enable Camera", value=False)
        
        # Recognition settings
        st.subheader("⚙️ Settings")
        threshold = st.slider("Recognition Threshold", 0.1, 1.0, THRESHOLD, 0.01)
        
        # Gloss buffer controls
        st.subheader("📝 Gloss Buffer")
        st.write(f"Current glosses: **{len(st.session_state.glosses_buffer)}**")
        
        if st.button("🗑️ Clear Glosses"):
            st.session_state.glosses_buffer = []
            st.success("Glosses cleared!")
        
        # Generate sentence button
        if st.button("🤖 Generate Sentence", disabled=len(st.session_state.glosses_buffer) == 0):
            if st.session_state.glosses_buffer:
                with st.spinner("Generating sentence with Gemini..."):
                    sentence = send_gemini_request(st.session_state.glosses_buffer)
                    st.session_state.generated_sentences.append({
                        "glosses": " ".join(st.session_state.glosses_buffer),
                        "sentence": sentence,
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                    st.session_state.glosses_buffer = []  # Clear buffer after generating
                st.success("Sentence generated!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Webcam Feed")
        
        if camera_enabled:
            # Camera input
            camera_input = st.camera_input("Take a picture")
            
            if camera_input is not None:
                # Convert to OpenCV format
                image = Image.open(camera_input)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Process frame for recognition
                recognized_gloss = process_frame_for_recognition(frame)
                
                # Display current recognition
                if st.session_state.confirmed_gloss_text:
                    st.success(f"✅ **Recognized:** {st.session_state.confirmed_gloss_text}")
                
                # Show frame status
                if len(st.session_state.frame_buffer) < CLIP_LEN:
                    st.info(f"Collecting frames... ({len(st.session_state.frame_buffer)}/{CLIP_LEN})")
        else:
            st.info("Enable camera to start recognition")
    
    with col2:
        st.subheader("📝 Current Glosses")
        
        if st.session_state.glosses_buffer:
            for i, gloss in enumerate(st.session_state.glosses_buffer, 1):
                st.write(f"{i}. {gloss}")
        else:
            st.write("No glosses collected yet")
    
    # Generated sentences section
    if st.session_state.generated_sentences:
        st.markdown("---")
        st.subheader("🤖 Generated Sentences")
        
        for i, result in enumerate(reversed(st.session_state.generated_sentences[-5:]), 1):
            with st.expander(f"Sentence {len(st.session_state.generated_sentences) - i + 1} ({result['timestamp']})"):
                st.write(f"**Glosses:** {result['glosses']}")
                st.write(f"**Sentence:** {result['sentence']}")
    
    # Instructions
    st.markdown("---")
    st.subheader("ℹ️ How to Use")
    st.markdown("""
    1. **Enable the camera** using the checkbox in the sidebar
    2. **Perform sign language** in front of the camera
    3. **Watch glosses** appear in the right column as they are recognized
    4. **Generate sentence** by clicking the button when you have enough glosses
    5. **View generated sentences** in the bottom section
    
    **Tips:**
    - Make sure you have good lighting
    - Keep your hands clearly visible
    - Wait for the frame buffer to fill up before starting
    - Adjust the recognition threshold if needed
    """)

if __name__ == '__main__':
    main()