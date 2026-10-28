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

# Verified fallback Retina & Eye Hospitals in Bengaluru if offline/rate-limited
VERIFIED_BENGALURU_CENTERS = [
    {
        "name": "Dr Agarwals Eye Hospital (Rajarajeshwari Nagar)",
        "specialty": "Ophthalmology & Diabetic Retinopathy Screening",
        "address": "1st Floor, Plot No 638, 80 Feet Road, Ideal Homes Layout, RR Nagar, Bengaluru 560098",
        "phone": "095949 24292",
        "lat": 12.9265,
        "lon": 77.5152,
    },
    {
        "name": "Sri Venkateshwara Nethralaya Eye Hospital",
        "specialty": "Vitreoretinal & Comprehensive Eye Care",
        "address": "120, 9th Main, Opp. Nimishamba Temple, Ideal Homes Twp, RR Nagar, Bengaluru 560098",
        "phone": "093795 12374",
        "lat": 12.9239,
        "lon": 77.5189,
    },
    {
        "name": "Retina Institute of Karnataka (The Eye Foundation)",
        "specialty": "Sub-Specialty Vitreoretinal & Diabetic Eye Institute",
        "address": "122, 5th Main Rd, Chamrajpet, Bengaluru, Karnataka 560018",
        "phone": "+91 80 2661 8661",
        "lat": 12.9579,
        "lon": 77.5663,
    },
    {
        "name": "Narayana Nethralaya Super Specialty Eye Hospital",
        "specialty": "Tertiary Vitreoretinal Surgery & OCT Diagnostics",
        "address": "121/C, Chord Rd, near ISKCON Temple, Rajajinagar, Bengaluru 560010",
        "phone": "080 6612 1641",
        "lat": 13.0068,
        "lon": 77.5506,
    },
    {
        "name": "Narayana Nethralaya (Bannerghatta Road)",
        "specialty": "Super Specialty Retina & Ophthalmic Referral Center",
        "address": "No. 63, Bannerghatta Rd, next to Royal Meenakshi Mall, Hulimavu, Bengaluru 560076",
        "phone": "080 6122 2400",
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
    Finds real Ophthalmologists & Eye Hospitals near the user's coordinates
    using IP geolocation fallback + OpenStreetMap Overpass API + verified clinical centers.
    """
    payload = request.get_json(silent=True) or {}
    lat = payload.get("lat")
    lon = payload.get("lon")
    city = "Bengaluru, Karnataka"

    # 1. If browser didn't send GPS coordinates, resolve via IP geolocation
    if lat is None or lon is None:
        try:
            req = urllib.request.Request(
                "https://ipapi.co/json/",
                headers={"User-Agent": "AetherRetinalWorkstation/1.0"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                ip_data = json.loads(resp.read().decode("utf-8"))
                lat = float(ip_data.get("latitude", 12.9250))
                lon = float(ip_data.get("longitude", 77.5200))
                city = f"{ip_data.get('city', 'Bengaluru')}, {ip_data.get('region', 'Karnataka')}"
        except Exception:
            lat, lon = 12.9250, 77.5200

    practitioners = []

    # 2. Query OpenStreetMap Overpass API for real eye hospitals & ophthalmologists within 12km
    try:
        overpass_query = f"""
        [out:json][timeout:6];
        (
          node["healthcare:speciality"~"ophthalmology"](around:12000,{lat},{lon});
          way["healthcare:speciality"~"ophthalmology"](around:12000,{lat},{lon});
          node["name"~"Eye Hospital|Nethralaya|Retina|Ophthal",i](around:12000,{lat},{lon});
          way["name"~"Eye Hospital|Nethralaya|Retina|Ophthal",i](around:12000,{lat},{lon});
        );
        out center 10;
        """
        data = urllib.parse.urlencode({"data": overpass_query}).encode("utf-8")
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=data,
            headers={"User-Agent": "AetherRetinalWorkstation/1.0"}
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            osm_json = json.loads(resp.read().decode("utf-8"))
            seen_names = set()
            for el in osm_json.get("elements", []):
                tags = el.get("tags", {})
                name = tags.get("name")
                if not name or name.lower() in seen_names:
                    continue
                seen_names.add(name.lower())
                el_lat = el.get("lat") or el.get("center", {}).get("lat")
                el_lon = el.get("lon") or el.get("center", {}).get("lon")
                if not el_lat or not el_lon:
                    continue

                dist = haversine_km(lat, lon, el_lat, el_lon)
                street = tags.get("addr:street", "")
                suburb = tags.get("addr:suburb", tags.get("addr:city", city))
                address = ", ".join(filter(None, [street, suburb])) or city
                phone = tags.get("phone", tags.get("contact:phone", ""))
                maps_q = urllib.parse.quote_plus(f"{name} {address}")

                practitioners.append({
                    "name": name,
                    "specialty": "Ophthalmology & Retinal Care Center",
                    "address": address,
                    "phone": phone,
                    "distance_km": dist,
                    "maps_url": f"https://www.google.com/maps/search/?api=1&query={maps_q}"
                })
    except Exception:
        pass

    # 3. Merge/Fallback with verified retina centers sorted by distance
    if len(practitioners) < 4:
        existing = {p["name"].lower() for p in practitioners}
        for center in VERIFIED_BENGALURU_CENTERS:
            if center["name"].lower() not in existing:
                dist = haversine_km(lat, lon, center["lat"], center["lon"])
                maps_q = urllib.parse.quote_plus(f"{center['name']} {center['address']}")
                practitioners.append({
                    "name": center["name"],
                    "specialty": center["specialty"],
                    "address": center["address"],
                    "phone": center["phone"],
                    "distance_km": dist,
                    "maps_url": f"https://www.google.com/maps/search/?api=1&query={maps_q}"
                })

    practitioners.sort(key=lambda x: x.get("distance_km", 999))
    return jsonify({
        "city": city,
        "practitioners": practitioners[:6]
    })


if __name__ == "__main__":
    print("Starting AETHER Retinal Workstation at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)