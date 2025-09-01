## Setup Docker
Build image with CPU

```bash
# build
docker build -t t5_service:latest .

# chạy - map port 5001 ra host
docker run --rm -d --name t5_service -p 5001:5001 t5_service:latest
```
Build image with GPU

```bash
docker build -t t5_service:gpu -f Dockerfile.gpu .
# cần docker engine hỗ trợ GPUs (nvidia-container-toolkit)
docker run --rm -d --gpus all --name t5_service_gpu -p 5001:5001 t5_service:gpu
```

## RUN 

```bash
python -m venv .venv
.venv\Scripts\activate       # on Windows


pip install -r requirements.txt
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

python gem_infer.py
```