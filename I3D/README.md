Python 3.9.7
Create venv
cd I3D
pip install -r requirements.txt
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

#Inference
python gem_infer.py
