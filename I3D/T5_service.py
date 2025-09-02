#!/usr/bin/env python3
# T5_service.py
# Simple Flask service that accepts glosses and returns generated sentence(s).
# - Accepts POST /generate_sentence with JSON {"session_id": "...", "glosses": ["G1","G2",...], "reset": false}
# - Keeps per-session gloss history (append) and generates sentence from history.
# - Env:
#     MODEL_NAME (default: "t5-small")
#     MAX_HISTORY (default: 20)
#     MAX_GEN_LEN (default: 64)
#     NUM_BEAMS (default: 5)
#     DEVICE (optional: "cpu" or "cuda"; autodetect if unset)
# Example:
#   curl -X POST http://localhost:5001/generate_sentence -H "Content-Type: application/json" \
#     -d '{"session_id":"s1","glosses":["HELLO","LOVE"],"reset":false}'

import os
import threading
from flask import Flask, request, jsonify
import time
import logging

# Transformer imports
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("T5_service")

app = Flask(__name__)

# Config from env
MODEL_NAME = os.getenv("MODEL_NAME", "t5-small")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
MAX_GEN_LEN = int(os.getenv("MAX_GEN_LEN", "64"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "5"))
DEVICE_ENV = os.getenv("DEVICE", None)  # "cpu" or "cuda" if you want to force
PORT = int(os.getenv("PORT", "5001"))

# Choose device
if DEVICE_ENV:
    device = torch.device(DEVICE_ENV)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Session storage (in-memory). If you need persistent storage, replace with DB.
sessions = {}  # session_id -> list of gloss strings
sessions_lock = threading.Lock()

# Load model & tokenizer (global)
logger.info(f"Loading model/tokenizer: {MODEL_NAME} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    logger.info("Model loaded.")
except Exception as e:
    logger.exception("Failed to load model. Check MODEL_NAME and network/cache.")
    raise

def build_prompt_from_glosses(gloss_list):
    """
    Convert list of glosses into a single prompt that T5 can take.
    You can modify this prompt format depending on how you want T5 to behave.
    """
    # Join glosses with separator — keep it readable for model.
    # Example prompt: "glosses: HELLO | LOVE YOU | THANK YOU. Generate natural sentence:"
    joined = " | ".join(gloss_list)
    prompt = f"glosses: {joined} -> generate sentence:"
    return prompt

def generate_sentence_from_prompt(prompt, max_length=MAX_GEN_LEN, num_beams=NUM_BEAMS):
    """
    Generate using model; returns the decoded string.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.strip()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": str(device)})

@app.route("/generate_sentence", methods=["POST"])
def generate_sentence():
    """
    Expected JSON body:
    {
      "session_id": "string",         # optional; default "default"
      "glosses": ["G1","G2", ...],   # one or multiple glosses
      "reset": false                  # optional; if true, clear session before appending
    }

    Response:
    {
      "session_id": "...",
      "sentence": "...",
      "session_gloss_count": N,
      "prompt_used": "..."
    }
    """
    start_time = time.time()
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid_json", "detail": str(e)}), 400

    session_id = data.get("session_id", "default")
    glosses = data.get("glosses", [])
    if glosses is None:
        glosses = []
    if not isinstance(glosses, list):
        return jsonify({"error": "glosses_must_be_list"}), 400

    reset_flag = bool(data.get("reset", False))

    # Update session buffer in thread-safe way
    with sessions_lock:
        if reset_flag or session_id not in sessions:
            sessions[session_id] = []
        # append glosses, keep last MAX_HISTORY
        sessions[session_id].extend([str(g).strip() for g in glosses if g is not None and str(g).strip() != ""])
        if len(sessions[session_id]) > MAX_HISTORY:
            sessions[session_id] = sessions[session_id][-MAX_HISTORY:]

        history = list(sessions[session_id])  # copy for use below

    if len(history) == 0:
        return jsonify({"error": "no_glosses_in_session"}), 400

    prompt = build_prompt_from_glosses(history)

    try:
        sentence = generate_sentence_from_prompt(prompt)
    except Exception as e:
        logger.exception("Error during generation")
        return jsonify({"error": "generation_failed", "detail": str(e)}), 500

    elapsed = time.time() - start_time
    resp = {
        "session_id": session_id,
        "sentence": sentence,
        "session_gloss_count": len(history),
        "prompt_used": prompt,
        "elapsed_sec": round(elapsed, 3)
    }
    return jsonify(resp), 200

if __name__ == "__main__":
    # Run Flask
    logger.info(f"Starting T5_service on 0.0.0.0:{PORT} (model {MODEL_NAME})")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
