"""
serve.py
---------
Flask server that:
  - Serves index.html (the frontend)
  - POST /api/route      -> runs eco_route_engine and returns JSON + plot
  - POST /api/detect     -> YOLO vehicle detection on uploaded road image
  - GET  /api/plot       -> streams the saved route_compare PNG
  - GET  /api/default_points -> returns demo origin/destination

YOLO integration:
  - Upload a road image via /api/detect
  - Returns vehicle counts + emission estimate per km
  - The route endpoint accepts optional vehicle_override from YOLO results
  - If no image is uploaded, fallback random values are used automatically
"""

import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    import _io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Graceful Flask import ─────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, send_file, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("[ERROR] Flask not found.  Run:  pip install flask flask-cors")
    sys.exit(1)

BASE_DIR = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE_DIR))
CORS(app)

PLOT_PATH = str(BASE_DIR / "route_compare_kolkata.png")

# ── Default Kolkata demo points (within the 3 km network) ────────────────────
DEFAULT_ORIGIN = (22.5680, 88.3512)   # near Esplanade / BBD Bagh
DEFAULT_DEST   = (22.5756, 88.3697)   # near Sealdah Station

# ── YOLO model singleton (loaded once) ────────────────────────────────────────
_yolo_model = None

def _get_yolo_model():
    """Lazy-load the YOLO model."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        # Look for local model first
        model_paths = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"]
        for mp in model_paths:
            if Path(mp).exists():
                _yolo_model = YOLO(str(mp))
                print(f"[YOLO] Loaded model: {mp}")
                return _yolo_model
            full = BASE_DIR / mp
            if full.exists():
                _yolo_model = YOLO(str(full))
                print(f"[YOLO] Loaded model: {full}")
                return _yolo_model
        # Try default download
        _yolo_model = YOLO("yolov8n.pt")
        print("[YOLO] Downloaded and loaded yolov8n.pt")
        return _yolo_model
    except Exception as e:
        print(f"[YOLO] Could not load model: {e}")
        return None

# COCO vehicle class IDs
VEHICLE_COCO_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Emissions per vehicle type (g CO2 per km)
EMISSIONS_G_PER_KM = {
    "car":        120,
    "motorcycle":  72,
    "bus":        822,
    "truck":      900,
    "van":        200,
    "bicycle":      0,
}


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Accept a road image upload, run YOLO vehicle detection.
    Returns:
    {
      "vehicle_counts": {"car": 5, "truck": 2, ...},
      "total_vehicles": 7,
      "co2_per_km_g": 1960,
      "annotated_b64": "base64 encoded annotated image",
      "source": "yolo_detection"
    }
    If no image or YOLO fails, returns fallback random values.
    """
    try:
        # Check if image file was uploaded
        if "image" not in request.files:
            return _fallback_vehicle_response("No image uploaded")

        file = request.files["image"]
        if file.filename == "":
            return _fallback_vehicle_response("Empty filename")

        # Read image bytes
        img_bytes = file.read()
        if len(img_bytes) < 100:
            return _fallback_vehicle_response("Image too small")

        # Try YOLO detection
        model = _get_yolo_model()
        if model is None:
            return _fallback_vehicle_response("YOLO model not available")

        import cv2
        import numpy as np

        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return _fallback_vehicle_response("Could not decode image")

        # Run inference
        results = model(frame, verbose=False, conf=0.35)

        # Count vehicles
        vehicle_counts = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_COCO_IDS:
                label = VEHICLE_COCO_IDS[cls_id]
                vehicle_counts[label] = vehicle_counts.get(label, 0) + 1

        total = sum(vehicle_counts.values())

        # Calculate CO2 per km
        co2_per_km = sum(
            cnt * EMISSIONS_G_PER_KM.get(vtype, 120)
            for vtype, cnt in vehicle_counts.items()
        )

        # Create annotated image
        annotated = results[0].plot()
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buf).decode()

        return jsonify({
            "vehicle_counts": vehicle_counts,
            "total_vehicles": total,
            "co2_per_km_g":   co2_per_km,
            "annotated_b64":  annotated_b64,
            "source":         "yolo_detection",
            "message":        f"Detected {total} vehicles via YOLOv8"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _fallback_vehicle_response(f"Detection error: {str(e)}")


def _fallback_vehicle_response(reason: str):
    """Generate random realistic vehicle counts as fallback."""
    import random
    counts = {
        "car":        random.randint(20, 80),
        "motorcycle": random.randint(5, 30),
        "bus":        random.randint(1, 8),
        "truck":      random.randint(2, 12),
    }
    total = sum(counts.values())
    co2 = sum(cnt * EMISSIONS_G_PER_KM.get(v, 120) for v, cnt in counts.items())

    return jsonify({
        "vehicle_counts": counts,
        "total_vehicles": total,
        "co2_per_km_g":   co2,
        "annotated_b64":  None,
        "source":         "random_fallback",
        "message":        f"Using random values ({reason}). "
                          f"Upload a road image for real YOLO detection."
    })


@app.route("/api/route", methods=["POST"])
def api_route():
    """
    Expects JSON or multipart form body:
    {
      "origin_lat": float,  "origin_lon": float,
      "dest_lat":   float,  "dest_lon":   float,
      "hour":       int     (optional, default 8),
      "vehicle_override": int  (optional, from YOLO detection)
    }
    Returns JSON with stats, route coordinates, and base64 plot.
    """
    try:
        body = request.get_json(force=True) or {}
        olat = float(body.get("origin_lat", DEFAULT_ORIGIN[0]))
        olon = float(body.get("origin_lon", DEFAULT_ORIGIN[1]))
        dlat = float(body.get("dest_lat",   DEFAULT_DEST[0]))
        dlon = float(body.get("dest_lon",   DEFAULT_DEST[1]))
        hour = int(body.get("hour", 8))
        vehicle_override = body.get("vehicle_override", None)

        from eco_route_engine import run_eco_routing
        result = run_eco_routing(
            olat, olon, dlat, dlon,
            hour=hour,
            plot_path=PLOT_PATH,
            vehicle_override=vehicle_override,
        )

        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        # Embed plot as base64 so the frontend doesn't need a second request
        if Path(PLOT_PATH).exists():
            with open(PLOT_PATH, "rb") as f:
                result["plot_b64"] = base64.b64encode(f.read()).decode()

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/plot")
def api_plot():
    if not Path(PLOT_PATH).exists():
        return jsonify({"error": "No plot yet. Call /api/route first."}), 404
    return send_file(PLOT_PATH, mimetype="image/png")


@app.route("/api/default_points")
def api_default_points():
    """Return the default demo origin/destination for Kolkata."""
    return jsonify({
        "origin": {"lat": DEFAULT_ORIGIN[0], "lon": DEFAULT_ORIGIN[1],
                   "label": "Esplanade / BBD Bagh"},
        "destination": {"lat": DEFAULT_DEST[0],   "lon": DEFAULT_DEST[1],
                        "label": "Sealdah Station"},
        "city": "Kolkata",
        "network_bounds": {
            "min_lat": 22.5455, "max_lat": 22.5996,
            "min_lon": 88.3347, "max_lon": 88.3931,
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*55}")
    print(f"  Eco-Routing Server starting on http://localhost:{port}")
    print(f"  Open your browser at: http://localhost:{port}")
    print(f"  YOLO vehicle detection: POST /api/detect")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
