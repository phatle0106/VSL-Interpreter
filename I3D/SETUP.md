
## RUN 

```bash
python -m venv .venv
.venv\Scripts\activate       # on Windows


pip install -r requirements.txt
pip install -r requirements_micro.txt

python gem_infer.py

# Start your service first
python run.py

# In another terminal
ssh -R 80:localhost:5000 serveo.net
```