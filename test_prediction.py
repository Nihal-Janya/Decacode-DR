import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from gradio_client import Client

try:
    from gradio_client import handle_file
except ImportError:
    handle_file = None

SPACE_ID = "bhimrazy/diabetic-retinopathy-detection"


def preprocess_retinal_scan(image_path: str) -> str:
    """
    Crops dark rectangular borders around the circular fundus and normalizes
    low-light exposure so microaneurysms and exudates are accurately graded.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    mask = arr.max(axis=2) > 15
    if np.any(mask):
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img.crop((x0, y0, x1, y1))

    w, h = img.size
    side = max(w, h)
    square_img = Image.new("RGB", (side, side), (0, 0, 0))
    square_img.paste(img, ((side - w) // 2, (side - h) // 2))

    if side < 512:
        square_img = square_img.resize((512, 512), Image.Resampling.LANCZOS)

    cropped_arr = np.array(square_img)
    if float(np.mean(cropped_arr)) < 65.0:
        square_img = ImageOps.autocontrast(square_img, cutoff=1)
        square_img = ImageEnhance.Contrast(square_img).enhance(1.25)
        square_img = ImageEnhance.Sharpness(square_img).enhance(1.3)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    square_img.save(tmp_file.name, format="PNG")
    tmp_file.close()
    return tmp_file.name


def run_retinal_prediction(image_path: str):
    resolved_path = str(Path(image_path.strip('"')).resolve())

    if not Path(resolved_path).is_file():
        raise FileNotFoundError(f"Error: Image file not found at {resolved_path}")

    preprocessed_path = preprocess_retinal_scan(resolved_path)

    try:
        demo = Client(SPACE_ID)
        file_input = (
            handle_file(preprocessed_path)
            if handle_file is not None
            else preprocessed_path
        )

        try:
            result = demo.predict(file_input, api_name="/predict")
        except Exception:
            result = demo.predict(file_input, fn_index=0)

        return result
    finally:
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)


if __name__ == "__main__":
    user_input = input("Enter the full path to your retinal image: ")
    try:
        output = run_retinal_prediction(user_input)
        print("\nPrediction result:")
        print(output)
    except Exception as e:
        print(e)