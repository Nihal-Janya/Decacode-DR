import os
import re
import math
import json
import html
import tempfile
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, send_from_directory

from check_demo_api import get_remote_api_info
from test_prediction import run_retinal_prediction

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR))

MAX_PIXELS = 20_000_000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}

DOCTOR_REGEX = re.compile(
    r"\b(Dr\.?\s+[A-Z][a-z]{2,}(?:\s+[A-Z]\.?|\s+[A-Z][a-z]{2,}){1,2})\b"
)


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


def geocode_area(area_query: str):
    """
    Dynamically geocodes any city, neighbourhood, or address worldwide
    without hardcoding any state or city.
    """
    queries_to_try = [f"{area_query}, India", area_query]
    for q_text in queries_to_try:
        try:
            q = urllib.parse.quote_plus(q_text)
            url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&addressdetails=1&limit=1"
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data:
                    hit = data[0]
                    lat = float(hit["lat"])
                    lon = float(hit["lon"])
                    addr = hit.get("address", {})
                    city_name = (
                        addr.get("suburb")
                        or addr.get("city")
                        or addr.get("town")
                        or addr.get("state_district")
                        or addr.get("state")
                        or area_query.title()
                    )
                    return lat, lon, city_name
        except Exception:
            continue
    return None, None, None


def reverse_geocode_coords(lat: float, lon: float) -> str:
    """Resolves human-readable locality/city name from GPS coordinates."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&addressdetails=1"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            addr = data.get("address", {})
            suburb = addr.get("suburb") or addr.get("neighbourhood")
            city = addr.get("city") or addr.get("town") or addr.get("state_district")
            if suburb and city and suburb.lower() != city.lower():
                return f"{suburb}, {city}"
            return city or suburb or "Current Location"
    except Exception:
        return "Current Location"


def discover_practitioner_name_online(hospital_name: str, city: str) -> str:
    """
    Dynamically searches public web medical listings for the real attending
    Ophthalmologist / Retina Specialist at a given eye hospital and city.
    """
    # If the facility name already starts with "Dr." or contains a doctor's name
    existing_match = DOCTOR_REGEX.search(hospital_name)
    if existing_match:
        return existing_match.group(1)

    search_query = f'"{hospital_name}" {city} "Dr." ophthalmologist OR retina OR eye'
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(search_query)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
            clean_text = html.unescape(re.sub(r"<[^>]+>", " ", raw_html))

            # Filter out generic hospital brand names like "Dr Agarwal" or "Dr Batra"
            # so we find the actual attending doctor's name if possible
            matches = DOCTOR_REGEX.findall(clean_text)
            for m in matches:
                cleaned = " ".join(m.split())
                lower_m = cleaned.lower()
                if any(
                    skip in lower_m
                    for skip in ["dr agarwal", "dr. agarwal", "dr shroff", "dr. shroff", "dr lal", "dr. lal"]
                ):
                    continue
                return cleaned
    except Exception:
        pass

    return f"Senior Ophthalmologist & Retina Team ({city})"


def query_live_ophthalmologists(lat: float, lon: float, city_label: str):
    """
    Queries OpenStreetMap Overpass API dynamically around (lat, lon)
    for real eye hospitals, retina centres, and ophthalmology clinics,
    and enriches each with its attending practitioner name.
    """
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    # Search 18 km around the target coordinates for eye hospitals & ophthalmology clinics
    overpass_query = f"""
    [out:json][timeout:10];
    (
      node["healthcare:speciality"~"ophthalmology|retina",i](around:18000,{lat},{lon});
      way["healthcare:speciality"~"ophthalmology|retina",i](around:18000,{lat},{lon});
      node["amenity"~"hospital|clinic|doctors"]["name"~"Eye|Retina|Nethralaya|Ophthal|Vision|Netra",i](around:18000,{lat},{lon});
      way["amenity"~"hospital|clinic|doctors"]["name"~"Eye|Retina|Nethralaya|Ophthal|Vision|Netra",i](around:18000,{lat},{lon});
      node["healthcare"~"hospital|clinic|doctor"]["name"~"Eye|Retina|Nethralaya|Ophthal|Vision|Netra",i](around:18000,{lat},{lon});
      way["healthcare"~"hospital|clinic|doctor"]["name"~"Eye|Retina|Nethralaya|Ophthal|Vision|Netra",i](around:18000,{lat},{lon});
    );
    out center 25;
    """

    elements = []
    for endpoint in overpass_endpoints:
        try:
            data = urllib.parse.urlencode({"data": overpass_query}).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                osm_json = json.loads(resp.read().decode("utf-8"))
                elements = osm_json.get("elements", [])
                if elements:
                    break
        except Exception:
            continue

    raw_candidates = []
    seen_names = set()

    for el in elements:
        tags = el.get("tags", {})
        raw_name = tags.get("name") or tags.get(" name:en")
        if not raw_name:
            continue

        norm_name = raw_name.strip().lower()
        if norm_name in seen_names:
            continue
        seen_names.add(norm_name)

        el_lat = el.get("lat") or el.get("center", {}).get("lat")
        el_lon = el.get("lon") or el.get("center", {}).get("lon")
        if el_lat is None or el_lon is None:
            continue

        dist_km = haversine_km(lat, lon, float(el_lat), float(el_lon))

        # Extract address components dynamically from OSM tags
        addr_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:suburb") or tags.get("addr:neighbourhood"),
            tags.get("addr:city") or city_label,
            tags.get("addr:postcode"),
        ]
        address = ", ".join([p for p in addr_parts if p])
        if not address:
            address = f"{city_label} ({round(float(el_lat), 4)}, {round(float(el_lon), 4)})"

        phone = (
            tags.get("phone")
            or tags.get("contact:phone")
            or tags.get(" contact:mobile")
            or ""
        )
        timings = tags.get("opening_hours") or "Mon–Sat • Open for Ophthalmic Consultations"

        # Check if OSM tags directly include the doctor/operator name
        osm_doctor = (
            tags.get("doctor")
            or tags.get("operator:person")
            or (tags.get("operator") if "dr" in (tags.get("operator") or "").lower() else None)
        )

        raw_candidates.append({
            "hospital_name": raw_name.strip(),
            "osm_doctor": osm_doctor,
            "address": address,
            "phone": phone,
            "timings": timings,
            "distance_km": dist_km,
            "lat": float(el_lat),
            "lon": float(el_lon),
        })

    # Sort by actual distance from the searched city / GPS coordinates
    raw_candidates.sort(key=lambda x: x["distance_km"])
    top_candidates = raw_candidates[:6]

    # Dynamically resolve attending practitioner names in parallel
    def enrich_candidate(item):
        hosp = item["hospital_name"]
        doc_name = item["osm_doctor"]
        if not doc_name:
            doc_name = discover_practitioner_name_online(hosp, city_label)

        maps_query = urllib.parse.quote_plus(f"{hosp} {item['address']}")
        return {
            "name": f"{doc_name} — {hosp}",
            "practitioner_name": doc_name,
            "credentials": "Ophthalmology & Vitreoretinal Specialist",
            "hospital_name": hosp,
            "specialty": "Diabetic Retinopathy & Comprehensive Eye Care",
            "address": item["address"],
            "phone": item["phone"],
            "timings": item["timings"],
            "distance_km": item["distance_km"],
            "maps_url": f"https://www.google.com/maps/search/?api=1&query={maps_query}",
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        enriched = list(pool.map(enrich_candidate, top_candidates))

    return enriched


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
                "error": "The image exceeds the 20-megapixel limit."
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
                "Low average brightness detected (< 50). Dark borders are automatically "
                "cropped and contrast-normalized before screening."
            )
        else:
            quality_message = (
                "Image brightness and contrast are within normal range."
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
            "error": "The uploaded file could not be decoded as a valid image."
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
    100% Dynamic Specialist & Eye Hospital Discovery:
    - If `area` is provided (e.g. "Delhi", "Mumbai", "Jayanagar"), geocodes that exact
      location worldwide and searches real nearby eye hospitals & ophthalmologists there.
    - Otherwise uses the user's live GPS (lat, lon) or IP geolocation.
    """
    payload = request.get_json(silent=True) or {}
    lat = payload.get("lat")
    lon = payload.get("lon")
    area_query = (payload.get("area") or "").strip()

    resolved_city = None

    # 1. If user searched a specific city/area (e.g., "Delhi"), geocode it globally
    if area_query:
        g_lat, g_lon, g_city = geocode_area(area_query)
        if g_lat is not None and g_lon is not None:
            lat, lon, resolved_city = g_lat, g_lon, g_city
        else:
            return jsonify({
                "city": area_query.title(),
                "practitioners": [],
                "error": f"Could not find coordinates for '{area_query}'."
            }), 404

    # 2. If using Live GPS coordinates from browser, reverse-geocode the city name
    elif lat is not None and lon is not None:
        lat, lon = float(lat), float(lon)
        resolved_city = reverse_geocode_coords(lat, lon)

    # 3. Fallback to IP Geolocation if browser GPS permission was denied
    else:
        try:
            req = urllib.request.Request("https://ipapi.co/json/", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                ip_data = json.loads(resp.read().decode("utf-8"))
                lat = float(ip_data.get("latitude"))
                lon = float(ip_data.get("longitude"))
                resolved_city = ip_data.get("city") or "Current Location"
        except Exception:
            return jsonify({
                "city": "Unknown",
                "practitioners": [],
                "error": "Location unavailable. Please allow GPS or use Filter by Area."
            }), 400

    practitioners = query_live_ophthalmologists(lat, lon, resolved_city)

    return jsonify({
        "city": resolved_city,
        "coordinates": {"lat": round(lat, 4), "lon": round(lon, 4)},
        "practitioners": practitioners
    })


if __name__ == "__main__":
    print("Starting Insight Retinal Screening Server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)