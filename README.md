# FaceID — Siamese Face Verification

Simple face **verification** using a Siamese network (pairwise or embedding) + a tiny FastAPI demo.

## Quickstart
```bash
pip install -r requirements.txt
python -m uvicorn app_fastapi:app --reload
# open http://127.0.0.1:8000/docs and try POST /verify