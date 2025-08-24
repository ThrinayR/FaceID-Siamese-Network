
from fastapi import FastAPI, UploadFile, File, Query
import tensorflow as tf
import numpy as np
from pathlib import Path

class L1Dist(tf.keras.layers.Layer):
    def call(self, a, b): return tf.math.abs(a - b)

custom_objects = {"L1Dist": L1Dist}

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "siamesemodel.keras"  # change if your file name differs

M = tf.keras.models.load_model(MODEL_PATH.as_posix(), custom_objects=custom_objects)

IS_PAIRWISE = (len(M.inputs) == 2 and M.output_shape[-1] == 1)
try:
    H = int(M.inputs[0].shape[1]); W = int(M.inputs[0].shape[2])
except Exception:
    H, W = 100, 100

app = FastAPI(title="Face Verification API")

def decode_and_resize(img_bytes, size):
    x = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    x = tf.image.resize(x, size)
    x = tf.cast(x, tf.float32) / 255.0
    return x

def embed_one(img_bytes):
    x = decode_and_resize(img_bytes, (H, W))[None, ...]
    z = M(x, training=False).numpy()[0]
    z = z / (np.linalg.norm(z) + 1e-9)
    return z

@app.get("/")
def info():
    return {"model": MODEL_PATH.name, "pairwise_classifier": IS_PAIRWISE, "input_size": [H, W]}

@app.post("/verify")
async def verify(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    threshold: float = Query(0.8, description="Use your calibrated value after Step 6")
):
    a_bytes = await file_a.read()
    b_bytes = await file_b.read()

    if IS_PAIRWISE:
        xa = decode_and_resize(a_bytes, (H, W))[None, ...]
        xb = decode_and_resize(b_bytes, (H, W))[None, ...]
        prob = float(M.predict([xa, xb], verbose=0).ravel()[0])
        return {"type": "pairwise", "score": prob, "match": prob >= threshold, "threshold": threshold}
    else:
        ea = embed_one(a_bytes)
        eb = embed_one(b_bytes)
        sim = float(np.dot(ea, eb))
        score = (sim + 1.0) / 2.0
        return {"type": "embedding", "score": score, "match": score >= threshold, "threshold": threshold}
