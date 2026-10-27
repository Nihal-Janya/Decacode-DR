import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, send_from_directory

# Import your two companion scripts
from check_demo_api import get_remote_api_info
from test_prediction import run_retinal_prediction

app = Flask(__name__, static_folder=".")

MAX_PIXELS = 20_000_000


@app.route("/")
def serve_ui():
    """Serves the single-page Luxury Glassmorphism HTML interface."""
    return send_from_directory(".", "index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "online"})


@app.route("/api/inspect", methods=["POST"])
def inspect_retinal_image():
    """
    Executes the exact Pillow & NumPy image inspection logic
    from your original app.py and returns JSON to index.html.
    """
    if "file" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    try:
        # Read file size in bytes
        uploaded_file.seek(0, os.SEEK_END)
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)

        # Open and decode the uploaded image
        image = Image.open(uploaded_file)
        image.load()

        # Reject unexpectedly large images (> 20,000,000 pixels)
        if image.width * image.height > MAX_PIXELS:
            return jsonify({
                "error": "The image is too large for this prototype (exceeds 20,000,000 pixels)."
            }), 400

        img_format = image.format or "UNKNOWN"
        img_size = list(image.size)
        img_mode = image.mode

        # Convert the image to RGB and NumPy array
        rgb_image = image.convert("RGB")
        pixels = np.array(rgb_image)

        # Calculate basic image statistics
        brightness = float(np.mean(pixels))
        contrast = float(np.std(pixels))

        low_brightness = brightness < 50.0
        if low_brightness:
            quality_message = (
                "Low average brightness. Inspect the image manually; "
                "this is not a diagnostic result."
            )
        else:
            quality_message = (
                "The average brightness is not below the example threshold. "
                "This does not establish clinical image quality."
            )

        return jsonify({
            "file_name": uploaded_file.filename,
            "file_size": file_size,
            "format": img_format,
            "dimensions": img_size,
            "mode": img_mode,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "shape": list(pixels.shape),
            "low_brightness": low_brightness,
            "quality_message": quality_message,
            "disclaimer": "No AI model has been run in this step. No disease prediction has been made."
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "The uploaded file could not be decoded as a valid image. Please try another image."
        }), 400


@app.route("/api/check-api", methods=["GET"])
def check_api_route():
    """Triggers check_demo_api.py and returns the API schema output."""
    try:
        info_text = get_remote_api_info()
        return jsonify({"api_info": info_text})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_route():
    """Saves the uploaded image temporarily and passes it to test_prediction.py."""
    if "file" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    uploaded_file = request.files["file"]
    suffix = Path(uploaded_file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        prediction_result = run_retinal_prediction(tmp_path)
        return jsonify({"result": prediction_result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print("Starting Retinal Prototype Server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
