# Dockerfile (CPU)
FROM python:3.10-slim

# Hệ thống lib cần thiết (ffmpeg/opencv deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsm6 libxext6 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy service vào image
COPY T5_service.py /app/T5_service.py
COPY requirements_t5.txt /app/requirements_t5.txt

# Upgrade pip và cài dependencies
RUN pip install --upgrade pip

# Thủ thuật: cài torch CPU wheel (nếu bạn muốn CPU-only)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Cài các package còn lại
RUN pip install -r /app/requirements_t5.txt

ENV PORT=5001
EXPOSE 5001

# Chạy service — đảm bảo T5_service.py khởi động trên 0.0.0.0
CMD ["python", "T5_service.py"]
