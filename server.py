# server.py — Unified FastAPI Backend Bridge for index.html
# Connects app.py, check_demo_api.py, and test_prediction.py to this web hub.
# Run with: uvicorn server:app --reload --port 8000

import io
import os
import tempfile
import contextlib
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from gradio_client import Client
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AETHER Ocular Telemetry & Gradio Bridge")

# Enable CORS so the HTML frontend can call your Python scripts directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_Methods=["*"],
    allow_headers=["*"],
)

HF_SPACE_ID = "bhimrazy/diabetic-retinopathy-detection"
MAX_PIXELS = 20_000_000

@app.get("/api/health")
def health_check():
    return {"status": "online", "space": HF_SPACE_ID}

# 1. Replaces Streamlit UI in app.py with a JSON Telemetry Endpoint
@app.post("/api/analyze")
async def analyze_retinal_image(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file.")

    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail="Image exceeds 20MP limit.")

    rgb_image = image.convert("RGB")
    pixels = np.array(rgb_image)
    brightness = float(np.mean(pixels))
    contrast = float(np.std(pixels))

    return {
        "filename": file.filename,
        "filesize": len(raw_bytes),
        "format": image.format or "PNG",
        "dimensions": [image.width, image.height],
        "mode": image.mode,
        "shape": list(pixels.shape),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "low_brightness_warning": brightness < 50.0
    }

# 2. Wraps check_demo_api.py and returns the schema text to the browser
@app.get("/api/check-api")
def inspect_gradio_api():
    demo = Client(HF_SPACE_ID)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        demo.view_api()
    return {"space": HF_SPACE_ID, "schema_output": buffer.getvalue()}

# 3. Wraps test_prediction.py for both uploaded files and local paths
@app.post("/api/predict")
async def predict_retinopathy(file: UploadFile = File(...)):
    demo = Client(HF_SPACE_ID)
    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = demo.predict(tmp_path, api_name="/predict")
        return {"status": "success", "prediction": result}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)