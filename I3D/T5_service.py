from flask import Flask, request, jsonify
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
from threading import Lock
import os
from dotenv import load_dotenv
from datetime import datetime
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#================================ CONFIGURATION ===============================
MODEL_NAME = "t5-base"  # You can change to t5-small for faster processing or t5-large for better quality
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global variables for model state
model = None
tokenizer = None
model_lock = Lock()

# Session storage for context
sessions = {}
sessions_lock = Lock()

#================================ UTILITY FUNCTIONS =================================

class SessionState:
    """Store session context for T5 processing"""
    def __init__(self):
        self.previous_sentences = []
        self.last_activity = datetime.now()
        self.context_history = []

def get_or_create_session(session_id):
    """Get or create a session state"""
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = SessionState()
        sessions[session_id].last_activity = datetime.now()
        return sessions[session_id]

def cleanup_old_sessions():
    """Remove sessions that haven't been active for more than 30 minutes"""
    with sessions_lock:
        current_time = datetime.now()
        inactive_sessions = [
            sid for sid, session in sessions.items()
            if (current_time - session.last_activity).seconds > 1800  # 30 minutes
        ]
        for sid in inactive_sessions:
            del sessions[sid]
        if inactive_sessions:
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")

def preprocess_glosses(glosses):
    """Preprocess glosses for T5 input"""
    if not glosses:
        return ""
    
    # Join glosses and clean them
    gloss_text = " ".join(glosses).strip()
    
    # Remove extra spaces and normalize
    gloss_text = re.sub(r'\s+', ' ', gloss_text)
    
    # Convert to lowercase for consistency
    gloss_text = gloss_text.lower()
    
    return gloss_text

def create_t5_prompt(glosses, context_history=None):
    """Create a proper prompt for T5 model"""
    gloss_text = preprocess_glosses(glosses)
    
    if not gloss_text:
        return "translate ASL glosses to sentence: "
    
    # Create context-aware prompt
    if context_history and len(context_history) > 0:
        # Use recent context to improve coherence
        recent_context = " ".join(context_history[-3:])  # Last 3 sentences
        prompt = f"translate ASL glosses to sentence with context '{recent_context}': {gloss_text}"
    else:
        prompt = f"translate ASL glosses to sentence: {gloss_text}"
    
    return prompt

def postprocess_sentence(sentence):
    """Clean and format the generated sentence"""
    if not sentence:
        return ""
    
    # Remove extra spaces
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    # Capitalize first letter
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    
    # Add period if missing
    if sentence and not sentence.endswith(('.', '!', '?')):
        sentence += '.'
    
    return sentence

def load_t5_model():
    """Load T5 model and tokenizer"""
    try:
        logger.info(f"Loading T5 model: {MODEL_NAME} on {DEVICE}")
        
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        # Move model to appropriate device
        model.to(DEVICE)
        model.eval()
        
        logger.info(f"T5 model loaded successfully on {DEVICE}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading T5 model: {e}")
        raise

def generate_sentence_from_glosses(glosses, session_context=None):
    """Generate sentence from glosses using T5"""
    try:
        with model_lock:
            if not model or not tokenizer:
                logger.error("Model or tokenizer not loaded")
                return None
            
            # Create prompt
            prompt = create_t5_prompt(glosses, session_context)
            
            # Tokenize input
            inputs = tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True
            ).to(DEVICE)
            
            # Generate sentence
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=MAX_OUTPUT_LENGTH,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process
            sentence = postprocess_sentence(generated_text)
            
            logger.info(f"Generated sentence: '{sentence}' from glosses: {glosses}")
            return sentence
            
    except Exception as e:
        logger.error(f"Error generating sentence: {e}")
        return None

#================================ API ENDPOINTS =================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_name': MODEL_NAME,
        'device': DEVICE,
        'active_sessions': len(sessions),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    """Generate sentence from glosses"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        # Get required fields
        session_id = data.get('session_id', 'default')
        glosses = data.get('glosses', [])
        
        if not glosses:
            return jsonify({
                'session_id': session_id,
                'sentence': '',
                'message': 'No glosses provided',
                'timestamp': datetime.now().isoformat()
            })
        
        # Get session context
        session = get_or_create_session(session_id)
        
        # Generate sentence
        sentence = generate_sentence_from_glosses(glosses, session.context_history)
        
        if sentence:
            # Update session context
            session.previous_sentences.append(sentence)
            session.context_history.append(sentence)
            
            # Keep only recent history (last 10 sentences)
            if len(session.context_history) > 10:
                session.context_history = session.context_history[-10:]
            
            if len(session.previous_sentences) > 20:
                session.previous_sentences = session.previous_sentences[-20:]
            
            return jsonify({
                'session_id': session_id,
                'sentence': sentence,
                'glosses_input': glosses,
                'confidence': 0.85,  # Placeholder confidence score
                'context_sentences': len(session.context_history),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'session_id': session_id,
                'error': 'Failed to generate sentence',
                'glosses_input': glosses,
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        logger.error(f"Error in generate_sentence endpoint: {str(e)}")
        return jsonify({'error': f'Sentence generation failed: {str(e)}'}), 500

@app.route('/get_session_history', methods=['GET'])
def get_session_history():
    """Get sentence history for a session"""
    try:
        session_id = request.args.get('session_id', 'default')
        
        with sessions_lock:
            if session_id not in sessions:
                return jsonify({
                    'session_id': session_id,
                    'sentences': [],
                    'message': 'Session not found'
                })
            
            session = sessions[session_id]
            
            return jsonify({
                'session_id': session_id,
                'sentences': session.previous_sentences,
                'context_history': session.context_history,
                'sentence_count': len(session.previous_sentences),
                'last_activity': session.last_activity.isoformat(),
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        return jsonify({'error': f'Failed to get session history: {str(e)}'}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset a specific session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        with sessions_lock:
            if session_id in sessions:
                sessions[session_id] = SessionState()
                logger.info(f"Reset T5 session: {session_id}")
            else:
                # Create new session
                sessions[session_id] = SessionState()
        
        return jsonify({
            'message': f'T5 session {session_id} reset successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        return jsonify({'error': f'Reset failed: {str(e)}'}), 500

@app.route('/test_generation', methods=['POST'])
def test_generation():
    """Test endpoint for sentence generation"""
    try:
        data = request.get_json()
        glosses = data.get('glosses', [])
        
        if not glosses:
            glosses = ['hello', 'my', 'name', 'john']  # Default test glosses
        
        sentence = generate_sentence_from_glosses(glosses)
        
        return jsonify({
            'test_glosses': glosses,
            'generated_sentence': sentence,
            'model_name': MODEL_NAME,
            'device': DEVICE,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in test generation: {str(e)}")
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_name': MODEL_NAME,
        'device': DEVICE,
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'max_input_length': MAX_INPUT_LENGTH,
        'max_output_length': MAX_OUTPUT_LENGTH,
        'active_sessions': len(sessions),
        'timestamp': datetime.now().isoformat()
    })

def initialize_service():
    """Initialize the T5 service"""
    global model, tokenizer
    
    try:
        logger.info("Initializing T5 ASL-to-Sentence Generation Service...")
        model, tokenizer = load_t5_model()
        logger.info("T5 service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize T5 service: {e}")
        raise

# Cleanup task
@app.before_request
def before_request():
    """Cleanup old sessions before each request"""
    if request.endpoint and len(sessions) > 0:
        cleanup_old_sessions()

if __name__ == '__main__':
    # Initialize the service
    initialize_service()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5001)),
        debug=False,
        threaded=True
    )