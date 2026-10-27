# app.py (Refactored so image telemetry is reusable by both Streamlit & FastAPI)
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_PIXELS = 20_000_000

def compute_retinal_telemetry(file_obj, filename="specimen.png", filesize=0) -> dict:
    """Core PIL + NumPy inspection logic extracted from Streamlit UI."""
    image = Image.open(file_obj)
    image.load()

    if image.width * image.height > MAX_PIXELS:
        raise ValueError("The image is too large for this prototype (>20MP).")

    rgb_image = image.convert("RGB")
    pixels = np.array(rgb_image)
    brightness = float(np.mean(pixels))
    contrast = float(np.std(pixels))

    return {
        "filename": filename,
        "filesize": filesize,
        "format": image.format,
        "dimensions": image.size,
        "mode": image.mode,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "shape": pixels.shape,
        "low_brightness_warning": brightness < 50.0
    }