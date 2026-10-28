import os
import math
import json
import tempfile
import urllib.request
import urllib.parse
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, send_from_directory

from check_demo_api import get_remote_api_info
from test_prediction import run_retinal_prediction

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR))

MAX_PIXELS = 20_000_000

# Verified Ophthalmologists & Retina Practitioners paired with their Hospitals
VERIFIED_BENGALURU_PRACTITIONERS = [
    {
        "practitioner_name": "Dr. Vinay Patil",
        "credentials": "MBBS, MD (AIIMS New Delhi) • Senior Vitreoretinal & Cataract Surgeon",
        "hospital_name": "Dr Agarwals Eye Hospital (Rajarajeshwari Nagar)",
        "specialty": "Diabetic Retinopathy, Medical Retina & Micro-Incision Surgery",
        "address": "1st Floor, Plot No 638, 80 Feet Road, Ideal Homes Layout, RR Nagar, Bengaluru 560098",
        "area_tags": ["rajarajeshwari nagar", "rr nagar", "ideal homes", "kengeri", "mysore road", "bengaluru"],
        "phone": "095949 24292",
        "timings": "Mon–Sat 9:00 AM – 8:00 PM",
        "lat": 12.9265,
        "lon": 77.5152,
    },
    {
        "practitioner_name": "Dr. Prashanth R",
        "credentials": "MBBS, DOMS, Fellowship in Vitreo-Retina • Chief Retina Consultant",
        "hospital_name": "Sri Venkateshwara Nethralaya Eye Hospital",
        "specialty": "Vitreoretinal Surgery, Diabetic Macular Edema & Retinal Lasers",
        "address": "120, 9th Main, Opp. Nimishamba Temple, Ideal Homes Twp, RR Nagar, Bengaluru 560098",
        "area_tags": ["rajarajeshwari nagar", "rr nagar", "banashankari", "uttarahalli", "nagarbhavi", "bengaluru"],
        "phone": "093795 12374",
        "timings": "Mon–Sat 9:00 AM – 9:00 PM",
        "lat": 12.9239,
        "lon": 77.5189,
    },
    {
        "practitioner_name": "Dr. P. Mohan",
        "credentials": "MBBS, MS (Ophthalmology), FMRF (Sankara Nethralaya) • Senior Retina Specialist",
        "hospital_name": "Retina Institute of Karnataka (The Eye Foundation)",
        "specialty": "Sub-Specialty Vitreoretinal & Diabetic Eye Care Institute",
        "address": "122, 5th Main Rd, Chamrajpet, Near Basavanagudi & Jayanagar, Bengaluru 560018",
        "area_tags": ["chamrajpet", "basavanagudi", "jayanagar", "jp nagar", "chickpet", "vv puram", "bengaluru"],
        "phone": "+91 80 2661 8661",
        "timings": "Mon–Sat 9:00 AM – 7:00 PM",
        "lat": 12.9579,
        "lon": 77.5663,
    },
    {
        "practitioner_name": "Dr. Chaitra Jayadev",
        "credentials": "MBBS, DOMS, FMRF, PhD • Head of Vitreoretinal Services",
        "hospital_name": "Narayana Nethralaya Super Specialty Eye Hospital (Rajajinagar)",
        "specialty": "Tertiary Vitreoretinal Surgery, OCT Angiography & Diabetic Eye Disease",
        "address": "121/C, Chord Rd, near ISKCON Temple, Rajajinagar, Bengaluru 560010",
        "area_tags": ["rajajinagar", "malleshwaram", "yeshwanthpur", "vijayanagar", "mahalakshmi layout", "bengaluru"],
        "phone": "080 6612 1641",
        "timings": "Mon–Sat 8:00 AM – 8:00 PM",
        "lat": 13.0068,
        "lon": 77.5506,
    },
    {
        "practitioner_name": "Dr. Mahesh P. Shanmugam",
        "credentials": "MBBS, DO, FRCSEd, PhD • Head of Vitreoretinal & Ocular Oncology",
        "hospital_name": "Sankara Eye Hospital (Kundalahalli / Marathahalli)",
        "specialty": "Proliferative Diabetic Retinopathy & Complex Vitreoretinal Surgery",
        "address": "Varthur Main Rd, Kundalahalli Gate, Marathahalli, Bengaluru 560037",
        "area_tags": ["marathahalli", "whitefield", "indiranagar", "kundalahalli", "bellandur", "bengaluru"],
        "phone": "080 6903 8900",
        "timings": "Mon–Sat 8:00 AM – 7:00 PM",
        "lat": 12.9592,
        "lon": 77.7155,
    },
    {
        "practitioner_name": "Dr. Naresh Babu K",
        "credentials": "MBBS, MS (Ophthalmology), Fellowship in Vitreo-Retina • Senior Retina Surgeon",
        "hospital_name": "Narayana Nethralaya Super Specialty Hospital (Bannerghatta Road)",
        "specialty": "Vitreoretinal Disorders, Intravitreal Therapy & Macular Surgery",
        "address": "No. 63, Bannerghatta Rd, next to Royal Meenakshi Mall, Hulimavu, South Bengaluru 560076",
        "area_tags": ["bannerghatta", "jayanagar", "jp nagar", "btm", "hulimavu", "koramangala", "hsr layout", "bengaluru"],
        "phone": "080 6122 2400",
        "timings": "Mon–Sat 8:00 AM – 7:00 PM",
        "lat": 12.8762,
        "lon": 77.5955,
    },
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return round(r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


@app.route("/")
def serve_ui():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "online"})


@app.route("/api/inspect", methods=["POST"])
def inspect_retinal_image():
    if "file" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    try:
        uploaded_file.seek(0, os.SEEK_END)
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)

        image = Image.open(uploaded_file)
        image.load()

        if image.width * image.height > MAX_PIXELS:
            return jsonify({
                "error": "The image exceeds the 20-megapixel limit for this workstation."
            }), 400

        img_format = image.format or "IMAGE"
        img_size = list(image.size)
        img_mode = image.mode

        rgb_image = image.convert("RGB")
        pixels = np.array(rgb_image)

        brightness = float(np.mean(pixels))
        contrast = float(np.std(pixels))

        low_brightness = brightness < 50.0
        if low_brightness:
            quality_message = (
                "Low average luminance detected (< 50). Auto-cropping dark borders "
                "and applying contrast normalization prior to neural inference."
            )
        else:
            quality_message = (
                "Optical luminance is within standard operating range. "
                "Proceeding with automated neural grading."
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
            "quality_message": quality_message
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "The uploaded file could not be decoded as a valid retinal photograph."
        }), 400


@app.route("/api/predict", methods=["POST"])
def predict_route():
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


@app.route("/api/referrals", methods=["POST"])
def find_nearby_ophthalmologists():
    """
    Returns both Practitioner Name (Doctor + Credentials) and Hospital Name
    filtered by the user's entered city/area or live GPS coordinates.
    """
    payload = request.get_json(silent=True) or {}
    lat = payload.get("lat")
    lon = payload.get("lon")
    area_query = (payload.get("area") or "Bengaluru").strip()
    area_lower = area_query.lower()

    if lat is None or lon is None:
        try:
            q = urllib.parse.quote_plus(f"{area_query}, Karnataka, India")
            req = urllib.request.Request(
                f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1",
                headers={"User-Agent": "AetherOphthalmicAtelier/3.0"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                geo = json.loads(resp.read().decode("utf-8"))
                if geo:
                    lat = float(geo[0]["lat"])
                    lon = float(geo[0]["lon"])
        except Exception:
            pass

    if lat is None or lon is None:
        lat, lon = 12.9250, 77.5200

    results = []
    for doc in VERIFIED_BENGALURU_PRACTITIONERS:
        dist = haversine_km(lat, lon, doc["lat"], doc["lon"])
        area_match = any(tag in area_lower or area_lower in tag for tag in doc["area_tags"])
        sort_score = dist - (4.0 if area_match else 0.0)

        maps_q = urllib.parse.quote_plus(f"{doc['hospital_name']} {doc['address']}")
        results.append({
            "name": f"{doc['practitioner_name']} — {doc['hospital_name']}",
            "practitioner_name": doc["practitioner_name"],
            "credentials": doc["credentials"],
            "hospital_name": doc["hospital_name"],
            "specialty": doc["specialty"],
            "address": doc["address"],
            "phone": doc["phone"],
            "timings": doc["timings"],
            "distance_km": dist,
            "_sort": sort_score,
            "maps_url": f"https://www.google.com/maps/search/?api=1&query={maps_q}"
        })

    results.sort(key=lambda x: x["_sort"])
    return jsonify({
        "city": area_query.title(),
        "practitioners": results
    })


if __name__ == "__main__":
    print("Starting AETHER Private Ophthalmic Atelier at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)