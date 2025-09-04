
## RUN 

```bash
python -m venv .venv
.venv\Scripts\activate       # on Windows


pip install -r requirements.txt
pip install -r requirements_micro.txt
pip uninstall torch torchvision torchaudio 
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

python gem_infer.py

# Start your service first
python run.py

# In another terminal
ssh -R 80:localhost:5000 serveo.net
```