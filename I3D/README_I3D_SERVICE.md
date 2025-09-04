I3D ASL Microservice

Overview
- Exposes the I3D ASL model as a local FastAPI service.
- Maintains per-session frame buffers and voting to provide stable top-1 gloss.
- Matches gem_infer.py defaults: CLIP_LEN=64, STRIDE=5, THRESHOLD=0.605, VOTING_BAG_SIZE=6.

Endpoints
- POST `/process_frame`
  - Body: `{ "session_id": "string", "frame": "<base64 image or data URL>" }`
  - Response: `{ recognition, current_gloss, buffer_size, frame_counter, session_id, timestamp, error }`
- POST `/reset_session`
  - Body: `{ "session_id": "string" }`
- GET `/health`

Run locally
1) Install deps (in a virtualenv/conda):
   - `pip install -r I3D/requirements_service.txt`
   - Ensure project weights exist: `weights/rgb_imagenet.pt` and `checkpoint/nslt_100_005624_0.756.pt`
   - Ensure labels at `preprocess/wlasl_class_list.txt`
2) Start service:
   - Recommended (from I3D dir): `cd I3D && uvicorn i3d_service:app --host 127.0.0.1 --port 5000`

Smoke test
- With the service running:
  - `python I3D/smoke_test_service.py`
  - Or from I3D dir: `python smoke_test_service.py`
  - It prints `/health` and a few `/process_frame` responses with `buffer_size` and any `recognition`.

Integration
- `ASL_system_backend/server.js` uses `I3D_SERVICE_URL` (default http://localhost:5000) and calls:
  - `POST /process_frame` for each frame
  - `POST /reset_session`
  - `GET /health`

Deployment Topology Note
- If backend runs on Render and this service stays local, expose the local service securely (Tailscale/Cloudflare Tunnel/ngrok) and set `I3D_SERVICE_URL` accordingly on the backend.
