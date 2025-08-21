FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 espeak-ng git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# đổi từ app.py sang textToSpeech.py
COPY textToSpeech.py .

# Prefetch model để tránh download lúc chạy
RUN python - <<'PY'
from TTS.api import TTS
TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
print("Prefetched model ✅")
PY

RUN mkdir -p /out
VOLUME ["/out"]

ENTRYPOINT ["python", "textToSpeech.py"]
CMD ["--text", "Hello from Docker Coqui!", "--out", "/out/output.wav", "--no-emphasis"]
