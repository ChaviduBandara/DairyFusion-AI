import base64
import uuid
import traceback
import requests
import sys
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# ── Ensure the backend folder is on Python's path ────────────────────────────
# This fixes "module not found" errors when uvicorn is launched from a
# different working directory than where app.py lives.
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
os.chdir(_BACKEND_DIR)   # also set cwd so relative model paths work
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Import pipeline ──────────────────────────────────────────────────────────
FUSION_AVAILABLE      = False
VISUAL_ONLY_AVAILABLE = False
pipeline              = None

try:
    from final_inference_v2 import create_pipeline
    FUSION_AVAILABLE = True
    print("[IMPORT] final_inference_v2 found — full fusion mode")
except Exception as e:
    print(f"[IMPORT] final_inference_v2 failed: {type(e).__name__}: {e}")
    print("[IMPORT] Trying original final_inference...")

if not FUSION_AVAILABLE:
    try:
        from final_inference import Config as VisualConfig, FullPipelineInference
        VISUAL_ONLY_AVAILABLE = True
        print("[IMPORT] final_inference found — visual-only mode")
    except Exception as e:
        print(f"[IMPORT] final_inference failed: {type(e).__name__}: {e}")
        print("[IMPORT] No inference module found — demo mode only")

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DairyFusion AI API",
    description="Multimodal yoghurt defect detection — YOLO + CNN + Sensor + Late Fusion",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000",
                   "http://localhost:3001", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Runtime folders ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
OUTPUTS_DIR = RUNTIME_DIR / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Live sensor storage ───────────────────────────────────────────────────────
# This holds the most recent reading sent by the ESP32.
# Reset to defaults on startup. Overwritten every time ESP32 sends data.
latest_sensor = {
    "yoghurt_temp_c":       43.5,    # from DS18B20
    "factory_humidity_pct": 62.0,    # from HTU21D
    "factory_temp_c":       28.0,    # from HTU21D (ambient)
    "weather_temp_c":       29.0,    # from Open-Meteo (fetched automatically)
    "weather_humidity_pct": 65.0,    # from Open-Meteo
    "precipitation_mm":     0.0,     # from Open-Meteo
    "production_slot":      "morning",
    "last_updated":         None,    # timestamp of last ESP32 reading
    "esp32_connected":      False,   # True once ESP32 has sent at least one reading
}

# Open-Meteo config — Kegalle, Sri Lanka
WEATHER_LAT  = 6.9271
WEATHER_LON  = 79.8612


# ─────────────────────────────────────────────────────────────────────────────
# Weather fetch from Open-Meteo API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_weather():
    """
    Fetches current outdoor weather from Open-Meteo API (free, no API key needed).
    Updates latest_sensor with weather_temp_c, weather_humidity_pct, precipitation_mm.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={WEATHER_LAT}&longitude={WEATHER_LON}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation"
            f"&timezone=Asia%2FColombo"
        )
        response = requests.get(url, timeout=5)
        data     = response.json()
        current  = data.get("current", {})

        latest_sensor["weather_temp_c"]       = current.get("temperature_2m", 29.0)
        latest_sensor["weather_humidity_pct"] = current.get("relative_humidity_2m", 65.0)
        latest_sensor["precipitation_mm"]     = current.get("precipitation", 0.0)

        print(f"[Weather] Updated: {latest_sensor['weather_temp_c']}°C  "
              f"{latest_sensor['weather_humidity_pct']}%  "
              f"{latest_sensor['precipitation_mm']}mm rain")

    except Exception as e:
        print(f"[Weather] Fetch failed: {e} — using last known values")


def get_production_slot() -> str:
    """Returns morning / midday / evening based on current hour."""
    hour = datetime.now().hour
    if 9 <= hour < 12:
        return "morning"
    elif 12 <= hour < 15:
        return "midday"
    elif 16 <= hour < 19:
        return "evening"
    else:
        return "morning"    # default outside production hours


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global pipeline

    # Load ML pipeline
    if FUSION_AVAILABLE:
        try:
            print("[STARTUP] Loading full DairyFusion pipeline...")
            pipeline = create_pipeline(
                sensor_model_dir      = str(BASE_DIR / "sensor_classifier_output"),
                yolo_model_path       = str(BASE_DIR / "dairynet_output_yolo_v7/train/weights/best.pt"),
                multimodal_model_path = str(BASE_DIR / "dairynet_multimodal_v4/best_multimodal.pt"),
            )
            print("[STARTUP] Full pipeline loaded.")
        except Exception as e:
            print(f"[STARTUP] Full pipeline failed: {e}")

    if pipeline is None and VISUAL_ONLY_AVAILABLE:
        try:
            print("[STARTUP] Loading visual-only pipeline...")
            pipeline = FullPipelineInference(VisualConfig())
            print("[STARTUP] Visual-only pipeline loaded.")
        except Exception as e:
            print(f"[STARTUP] Visual-only pipeline failed: {e}")

    if pipeline is None:
        print("[STARTUP] No models loaded — demo mode active")

    # Fetch initial weather
    fetch_weather()
    latest_sensor["production_slot"] = get_production_slot()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def _demo_result(image: np.ndarray) -> dict:
    import random
    rng     = random.Random(42)
    defects = ["foam", "water_droplets", "residue", "water_layer", "none"]
    chosen  = rng.choice(defects)
    has_def = chosen != "none"
    ROOT = {
        "foam":          "Yoghurt temperature exceeded filling range (>44.8°C)",
        "water_droplets":"High factory humidity + low yoghurt temp causing condensation",
        "residue":       "Hot and dry conditions causing surface residue",
        "water_layer":   "Yoghurt temp outside stable range causing syneresis",
        "none":          "No defect detected — all parameters within normal range",
    }
    ACT = {
        "foam":          "STOP LINE → Check heater calibration",
        "water_droplets":"ALERT → Reduce factory humidity → Check ventilation",
        "residue":       "MONITOR → Clean filling nozzles",
        "water_layer":   "ALERT → Adjust temperature control",
        "none":          "OK — No action required",
    }
    h, w = image.shape[:2]
    out  = image.copy()
    cv2.rectangle(out, (0,0), (w,60), (0,140,0) if not has_def else (0,0,180), -1)
    cv2.putText(out, f"[{'DEFECT' if has_def else 'CLEAN'}] {chosen.upper()} — DEMO",
                (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(out, ROOT[chosen][:72], (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)
    conf = round(rng.uniform(72, 95), 1)
    return {
        "ok": True, "demo_mode": True,
        "defect_detected":   has_def,
        "final_defect":      chosen,
        "final_confidence":  conf,
        "final_severity":    "none" if not has_def else rng.choice(["low","medium","high"]),
        "root_cause":        ROOT[chosen],
        "action":            ACT[chosen],
        "fusion_mode":       "Demo mode — real models not loaded",
        "visual_class":      chosen if has_def else "cup",
        "visual_confidence": round(rng.uniform(65,92), 1),
        "sensor_type":       chosen if has_def else "none",
        "sensor_confidence": round(rng.uniform(60,88), 1),
        "sensor_severity":   "none" if not has_def else rng.choice(["low","medium"]),
        "sensor_used":       True,
        "prediction":        chosen,
        "confidence":        round(conf/100, 4),
        "method":            "Demo",
        "detections":        [{"class_name":chosen,"conf":0.85,"box":[50,50,200,200]}] if has_def else [],
        "_image":            out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model for ESP32 POST body
# ─────────────────────────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    yoghurt_temp_c:       float          # DS18B20 — yoghurt temperature
    factory_humidity_pct: float          # HTU21D  — factory air humidity
    factory_temp_c:       Optional[float] = None   # HTU21D — factory air temp


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
def health():
    mode = "demo"
    if pipeline is not None:
        mode = "fusion" if FUSION_AVAILABLE else "visual_only"
    return {
        "ok":             True,
        "pipeline_loaded":pipeline is not None,
        "mode":           mode,
        "demo_mode":      pipeline is None,
        "fusion_mode":    FUSION_AVAILABLE and pipeline is not None,
        "esp32_connected":latest_sensor["esp32_connected"],
    }


# ── ESP32 pushes readings here ────────────────────────────────────────────────
@app.post("/api/sensor")
async def receive_sensor(reading: SensorReading):
    """
    Called by the ESP32 every 5 seconds.
    Updates the stored sensor values and refreshes weather data.
    """
    global latest_sensor

    # Update from ESP32
    latest_sensor["yoghurt_temp_c"]       = round(reading.yoghurt_temp_c, 2)
    latest_sensor["factory_humidity_pct"] = round(reading.factory_humidity_pct, 2)
    if reading.factory_temp_c is not None:
        latest_sensor["factory_temp_c"]   = round(reading.factory_temp_c, 2)
    latest_sensor["last_updated"]         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    latest_sensor["esp32_connected"]      = True
    latest_sensor["production_slot"]      = get_production_slot()

    # Refresh outdoor weather from Open-Meteo
    fetch_weather()

    print(f"[ESP32] Received → yoghurt={latest_sensor['yoghurt_temp_c']}°C  "
          f"humidity={latest_sensor['factory_humidity_pct']}%  "
          f"time={latest_sensor['last_updated']}")

    return {
        "ok":       True,
        "received": {
            "yoghurt_temp_c":       latest_sensor["yoghurt_temp_c"],
            "factory_humidity_pct": latest_sensor["factory_humidity_pct"],
            "weather_temp_c":       latest_sensor["weather_temp_c"],
        }
    }


# ── Frontend reads latest sensor state ───────────────────────────────────────
@app.get("/api/sensor/latest")
def get_latest_sensor():
    """
    Frontend polls this every few seconds to show live sensor readings.
    """
    return {
        "ok":                 True,
        "yoghurt_temp_c":     latest_sensor["yoghurt_temp_c"],
        "factory_humidity_pct": latest_sensor["factory_humidity_pct"],
        "factory_temp_c":     latest_sensor["factory_temp_c"],
        "weather_temp_c":     latest_sensor["weather_temp_c"],
        "weather_humidity_pct": latest_sensor["weather_humidity_pct"],
        "precipitation_mm":   latest_sensor["precipitation_mm"],
        "production_slot":    latest_sensor["production_slot"],
        "last_updated":       latest_sensor["last_updated"],
        "esp32_connected":    latest_sensor["esp32_connected"],
    }


@app.get("/api/sensor-defaults")
def sensor_defaults():
    return {
        "yoghurt_temp_c":       latest_sensor["yoghurt_temp_c"],
        "factory_humidity_pct": latest_sensor["factory_humidity_pct"],
        "weather_temp_c":       latest_sensor["weather_temp_c"],
        "weather_humidity_pct": latest_sensor["weather_humidity_pct"],
        "precipitation_mm":     latest_sensor["precipitation_mm"],
        "production_slot":      latest_sensor["production_slot"],
    }


# ── Main analysis endpoint ───────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    file:             UploadFile      = File(...),
    yoghurt_temp:     Optional[float] = Form(None),
    factory_humidity: Optional[float] = Form(None),
    weather_temp:     Optional[float] = Form(None),
    weather_humidity: Optional[float] = Form(None),
    precipitation:    Optional[float] = Form(None),
    production_slot:  Optional[str]   = Form(None),
    use_live_sensor:  Optional[bool]  = Form(True),   # if True, use ESP32 data automatically
):
    """
    Main analysis endpoint.

    If use_live_sensor=True (default) AND the ESP32 has sent at least one reading,
    the live sensor values are used automatically — no manual entry needed.

    Manual form values override live sensor values if provided.
    """
    try:
        # Save uploaded image
        suffix    = Path(file.filename).suffix.lower() or ".jpg"
        img_id    = str(uuid.uuid4())
        img_path  = UPLOADS_DIR / f"{img_id}{suffix}"
        raw_bytes = await file.read()
        img_path.write_bytes(raw_bytes)

        # ── Build sensor_data ──────────────────────────────────────────────
        # Priority: manual form values > live ESP32 > defaults
        if use_live_sensor and latest_sensor["esp32_connected"]:
            # Use real ESP32 data as base
            sensor_data = {
                "yoghurt_temp_c":       latest_sensor["yoghurt_temp_c"],
                "factory_humidity_pct": latest_sensor["factory_humidity_pct"],
                "weather_temp_c":       latest_sensor["weather_temp_c"],
                "weather_humidity_pct": latest_sensor["weather_humidity_pct"],
                "precipitation_mm":     latest_sensor["precipitation_mm"],
                "production_slot":      latest_sensor["production_slot"],
            }
            data_source = "live_esp32"
        else:
            # Use defaults
            sensor_data = {
                "yoghurt_temp_c":       43.5,
                "factory_humidity_pct": 62.0,
                "weather_temp_c":       latest_sensor["weather_temp_c"],
                "weather_humidity_pct": latest_sensor["weather_humidity_pct"],
                "precipitation_mm":     latest_sensor["precipitation_mm"],
                "production_slot":      get_production_slot(),
            }
            data_source = "defaults"

        # Manual overrides (if user typed in the frontend fields)
        if yoghurt_temp     is not None: sensor_data["yoghurt_temp_c"]       = yoghurt_temp
        if factory_humidity is not None: sensor_data["factory_humidity_pct"] = factory_humidity
        if weather_temp     is not None: sensor_data["weather_temp_c"]       = weather_temp
        if weather_humidity is not None: sensor_data["weather_humidity_pct"] = weather_humidity
        if precipitation    is not None: sensor_data["precipitation_mm"]     = precipitation
        if production_slot  is not None: sensor_data["production_slot"]      = production_slot

        print(f"[Analyze] Sensor source: {data_source}")
        print(f"[Analyze] yoghurt_temp={sensor_data['yoghurt_temp_c']}°C  "
              f"factory_humidity={sensor_data['factory_humidity_pct']}%")

        # ── Demo mode ──────────────────────────────────────────────────────
        if pipeline is None:
            arr  = np.frombuffer(raw_bytes, np.uint8)
            img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            demo = _demo_result(img if img is not None else np.zeros((400,400,3), np.uint8))
            out_img  = demo.pop("_image")
            out_path = OUTPUTS_DIR / f"{img_id}_annotated.jpg"
            cv2.imwrite(str(out_path), out_img)
            demo["annotated_image"]  = _to_b64(out_img)
            demo["sensor_source"]    = data_source
            demo["sensor_data_used"] = sensor_data
            demo["esp32_connected"]  = latest_sensor["esp32_connected"]
            return JSONResponse(demo)

        # ── Full fusion pipeline ───────────────────────────────────────────
        if FUSION_AVAILABLE:
            out_path = OUTPUTS_DIR / f"{img_id}_annotated.jpg"
            result   = pipeline.run(
                image_path  = str(img_path),
                sensor_data = sensor_data,
                output_path = str(out_path),
            )

            # Reject images that don't contain a yoghurt cup
            if result.get("not_a_cup"):
                return JSONResponse({
                    "ok":       False,
                    "not_a_cup": True,
                    "error":    "No yoghurt cup detected. Please upload an image of a yoghurt cup.",
                }, status_code=422)

            return JSONResponse({
                "ok":                True,
                "demo_mode":         False,
                "defect_detected":   result["defect_detected"],
                "final_defect":      result["final_defect"],
                "final_confidence":  result["final_confidence"],
                "final_severity":    result["final_severity"],
                "root_cause":        result["root_cause"],
                "action":            result["action"],
                "fusion_mode":       result["fusion_mode"],
                "timestamp":         result["timestamp"],
                "visual_class":      result["visual_class"],
                "visual_confidence": result["visual_confidence"],
                "sensor_type":       result["sensor_type"],
                "sensor_confidence": result["sensor_confidence"],
                "sensor_severity":   result["sensor_severity"],
                "sensor_used":       result["sensor_used"],
                "detections":        result.get("detections", []),
                "prediction":        result["final_defect"],
                "confidence":        result["final_confidence"] / 100,
                "method":            "DairyFusion (YOLO + CNN + Sensor + Late Fusion)",
                "annotated_image":   _to_b64(result["annotated_image"]),
                # Extra info for frontend
                "sensor_source":     data_source,
                "sensor_data_used":  sensor_data,
                "esp32_connected":   latest_sensor["esp32_connected"],
                "last_sensor_update":latest_sensor["last_updated"],
            })

        # ── Visual-only fallback ───────────────────────────────────────────
        result = pipeline.process_image(img_path)
        if result is None:
            return JSONResponse({"ok": False, "error": "Could not read image"}, status_code=400)

        out_path = OUTPUTS_DIR / f"{img_id}_annotated.jpg"
        cv2.imwrite(str(out_path), result["image"])
        pred       = result["prediction_name"]
        conf_pct   = round(result["confidence"] * 100, 1)
        has_defect = pred not in ("cup","none","no_defect","clean")

        return JSONResponse({
            "ok": True, "demo_mode": False,
            "defect_detected":   has_defect,
            "final_defect":      pred,
            "final_confidence":  conf_pct,
            "final_severity":    "medium" if has_defect else "none",
            "root_cause":        "Detected by YOLO + Multimodal CNN",
            "action":            "Inspect the production batch" if has_defect else "OK",
            "fusion_mode":       "Visual-only mode",
            "visual_class":      pred,
            "visual_confidence": conf_pct,
            "sensor_type":       "unavailable",
            "sensor_confidence": 0,
            "sensor_severity":   "unavailable",
            "sensor_used":       False,
            "detections":        result.get("detections", []),
            "prediction":        pred,
            "confidence":        result["confidence"],
            "method":            result.get("method","Visual"),
            "annotated_image":   _to_b64(result["image"]),
            "sensor_source":     data_source,
            "esp32_connected":   latest_sensor["esp32_connected"],
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
