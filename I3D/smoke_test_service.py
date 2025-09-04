import base64
import json
import os
import sys
import time
from io import BytesIO

import cv2
import numpy as np
import urllib.request


SERVICE_URL = os.environ.get("I3D_SERVICE_URL", "http://127.0.0.1:5000")


def encode_image_b64(img: np.ndarray) -> str:
    # BGR to JPEG
    ok, buf = cv2.imencode('.jpg', img)
    if not ok:
        raise RuntimeError('Failed to encode image')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def http_post_json(url: str, payload: dict, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def main():
    # Create a dummy frame (black image with a white square)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (220, 140), (420, 340), (255, 255, 255), -1)
    b64 = encode_image_b64(img)

    session_id = f"smoke_{int(time.time())}"
    print(f"Service: {SERVICE_URL}")
    print(f"Session: {session_id}")

    # Health check
    try:
        with urllib.request.urlopen(f"{SERVICE_URL}/health", timeout=5) as resp:
            print("Health:", resp.read().decode('utf-8'))
    except Exception as e:
        print("Health check failed:", e)
        sys.exit(1)

    # Post a few frames to see buffer behavior
    for i in range(5):
        out = http_post_json(f"{SERVICE_URL}/process_frame", {
            'session_id': session_id,
            'frame': b64,
        })
        print(f"Response {i+1}:", out)

    print("Done.")


if __name__ == '__main__':
    main()

