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

# --- phần config và import giữ nguyên như trước ---
import os
import threading
from flask import Flask, request, jsonify
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("T5_service")

app = Flask(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "t5-small")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))
MIN_GLOSSES = int(os.getenv("MIN_GLOSSES", "3"))   # <--- tối thiểu 3 gloss để generate
MAX_GEN_LEN = int(os.getenv("MAX_GEN_LEN", "64"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "5"))
DEVICE_ENV = os.getenv("DEVICE", None)
PORT = int(os.getenv("PORT", "5001"))

device = torch.device(DEVICE_ENV if DEVICE_ENV else ("cuda" if torch.cuda.is_available() else "cpu"))
logger.info(f"Using device: {device}")

sessions = {}
sessions_lock = threading.Lock()

logger.info(f"Loading model/tokenizer: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
logger.info("Model loaded.")

def build_prompt_from_glosses(gloss_list):
    joined = ", ".join(gloss_list)
    prompt = f"Convert the following sign-language glosses into a natural English sentence: {joined}"
    return prompt

def generate_sentence_from_prompt(prompt, max_length=MAX_GEN_LEN, num_beams=NUM_BEAMS):
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
    Body:
    {
      "session_id": "s1",
      "glosses": ["G1","G2",...],  # can be 1..n
      "reset": false
    }
    Behavior:
    - Append glosses to session buffer (up to MAX_HISTORY)
    - Only generate when session buffer has >= MIN_GLOSSES
    - If not enough, return generated=false with count info
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

    with sessions_lock:
        if reset_flag or session_id not in sessions:
            sessions[session_id] = []
        # append new glosses, keep only non-empty strings
        sessions[session_id].extend([str(g).strip() for g in glosses if g is not None and str(g).strip() != ""])
        # truncate to last MAX_HISTORY
        if len(sessions[session_id]) > MAX_HISTORY:
            sessions[session_id] = sessions[session_id][-MAX_HISTORY:]
        history = list(sessions[session_id])

    # If not enough glosses overall in session, do NOT generate
    if len(history) < MIN_GLOSSES:
        return jsonify({
            "session_id": session_id,
            "generated": False,
            "reason": "not_enough_glosses",
            "session_gloss_count": len(history),
            "min_glosses_required": MIN_GLOSSES
        }), 200

    # generate from history (or you could generate from last K only)
    prompt = build_prompt_from_glosses(history)
    try:
        sentence = generate_sentence_from_prompt(prompt)
    except Exception as e:
        logger.exception("Error during generation")
        return jsonify({"error": "generation_failed", "detail": str(e)}), 500

    elapsed = time.time() - start_time
    return jsonify({
        "session_id": session_id,
        "generated": True,
        "sentence": sentence,
        "session_gloss_count": len(history),
        "prompt_used": prompt,
        "elapsed_sec": round(elapsed, 3)
    }), 200

if __name__ == "__main__":
    logger.info(f"Starting T5_service on 0.0.0.0:{PORT} (model {MODEL_NAME})")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
