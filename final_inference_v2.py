"""
================================================================================
DairyFusion AI — Unified Final Inference Pipeline v2
================================================================================

Connects THREE components into one end-to-end system:

  STREAM 1 — Visual Model
    Camera image → DairyNet YOLO V7 (detection) +
                   DairyNet Multimodal V4 (classification)
    Output: defect class + confidence

  STREAM 2 — Sensor Classifier
    ESP32 sensor readings → Gradient Boosting / RandomForest classifier
    Output: defect_type + defect_severity

  LATE FUSION ENGINE
    Combines both streams at decision level
    Output: final defect, root cause, severity, action recommendation

USAGE:
  # Live mode (real ESP32 connected):
  python final_inference_v2.py --image cup.jpg --live-sensor

  # Manual sensor values:
  python final_inference_v2.py --image cup.jpg \
      --yoghurt-temp 41.5 --factory-humidity 85.0 \
      --weather-temp 27.0 --weather-humidity 88.0 \
      --precipitation 2.5 --slot evening

  # Image only (no sensor):
  python final_inference_v2.py --image cup.jpg
================================================================================
"""

import os
import json
import argparse
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Visual model paths ────────────────────────────────────────────────────
    YOLO_MODEL_PATH        = "dairynet_output_yolo_v7/train/weights/best.pt"
    MULTIMODAL_MODEL_PATH  = "dairynet_multimodal_v4/best_multimodal.pt"

    # ── Sensor classifier path ─────────────────────────────────────────────
    # Folder produced by dairyfusion_sensor_classifier_v2.py
    SENSOR_MODEL_DIR       = "sensor_classifier_output"

    # ── Visual model settings ──────────────────────────────────────────────
    CLASS_NAMES = [
        "cup", "dustparticle", "eyelash", "foam", "hair", "insect",
        "overfill", "plasticparticle", "residue", "underfill",
        "waterbubble", "waterlayer"
    ]
    NUM_CLASSES       = 12
    IMAGE_FEATURE_DIM = 256
    METADATA_FEATURE_DIM = 64
    HIDDEN_DIM        = 128
    DROPOUT           = 0.3
    CROP_SIZE         = 64

    OPTIMAL_THRESHOLDS = {
        "cup": 0.50, "dustparticle": 0.25, "eyelash": 0.20, "foam": 0.10,
        "hair": 0.30, "insect": 0.30, "overfill": 0.30, "plasticparticle": 0.25,
        "residue": 0.25, "underfill": 0.30, "waterbubble": 0.25, "waterlayer": 0.25,
    }
    BBOX_CORRECTION_CLASSES = [1, 2, 3, 5, 7, 8, 10]
    SEGMENT_CLASSES         = [0, 6, 9, 11]
    TYPICAL_DEFECT_SIZES    = {
        1: (40, 35), 2: (45, 40), 3: (60, 50), 5: (45, 40),
        7: (40, 35), 8: (50, 45), 10: (45, 40),
    }
    COLORS = {
        "cup": (255, 200, 0), "dustparticle": (0, 0, 255),
        "eyelash": (0, 255, 0), "foam": (255, 0, 0),
        "hair": (255, 0, 255), "insect": (255, 255, 0),
        "overfill": (0, 255, 128), "plasticparticle": (0, 128, 255),
        "residue": (128, 0, 255), "underfill": (255, 128, 0),
        "waterbubble": (0, 255, 255), "waterlayer": (128, 255, 0),
    }

    # ── Sensor default values (used when sensor unavailable) ──────────────
    DEFAULT_SENSOR = {
        "yoghurt_temp_c":       43.5,
        "factory_humidity_pct": 62.0,
        "weather_temp_c":       29.0,
        "weather_humidity_pct": 65.0,
        "precipitation_mm":     0.0,
        "production_slot":      "morning",
    }

    # ── Late fusion confidence threshold ──────────────────────────────────
    SENSOR_CONFIDENCE_THRESHOLD = 60.0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VISUAL MODEL COMPONENTS (unchanged from original)
# ═════════════════════════════════════════════════════════════════════════════

def correct_bbox(x1, y1, x2, y2, cls_id, img_w, img_h):
    if cls_id in Config.SEGMENT_CLASSES:
        return x1, y1, x2, y2
    if cls_id == 4:
        return x1, y1, x2, y2
    if cls_id in Config.BBOX_CORRECTION_CLASSES:
        actual_cx, actual_cy = x2, y2
        w, h = Config.TYPICAL_DEFECT_SIZES.get(cls_id, (40, 35))
        return (int(max(0, actual_cx - w/2)), int(max(0, actual_cy - h/2)),
                int(min(img_w, actual_cx + w/2)), int(min(img_h, actual_cy + h/2)))
    return x1, y1, x2, y2


class ROIFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class MultimodalFusionClassifier(nn.Module):
    def __init__(self, image_dim=256, metadata_dim=64, hidden_dim=128,
                 num_classes=12, dropout=0.3):
        super().__init__()
        self.image_proj    = nn.Sequential(nn.Linear(image_dim, hidden_dim),
                                           nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.metadata_proj = nn.Sequential(nn.Linear(metadata_dim, hidden_dim),
                                           nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.gate    = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
        self.fusion  = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                     nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier   = nn.Linear(hidden_dim, num_classes)
        self.image_aux    = nn.Linear(hidden_dim, num_classes)
        self.metadata_aux = nn.Linear(hidden_dim, num_classes)

    def forward(self, image_features, metadata_features, return_aux=False):
        img_proj  = self.image_proj(image_features)
        meta_proj = self.metadata_proj(metadata_features)
        attended, _ = self.cross_attention(img_proj.unsqueeze(1), meta_proj.unsqueeze(1), meta_proj.unsqueeze(1))
        attended = attended.squeeze(1)
        gate = self.gate(torch.cat([img_proj, meta_proj], dim=-1))
        fused = self.fusion(torch.cat([img_proj * gate + attended, meta_proj * (1 - gate)], dim=-1))
        logits = self.classifier(fused)
        if return_aux:
            return logits, self.image_aux(img_proj), self.metadata_aux(meta_proj)
        return logits


class DairyNetMultimodal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roi_extractor     = ROIFeatureExtractor(config.IMAGE_FEATURE_DIM)
        self.fusion_classifier = MultimodalFusionClassifier(
            config.IMAGE_FEATURE_DIM, config.METADATA_FEATURE_DIM,
            config.HIDDEN_DIM, config.NUM_CLASSES, config.DROPOUT)

    def forward(self, roi_images, metadata_features, return_aux=False):
        return self.fusion_classifier(self.roi_extractor(roi_images), metadata_features, return_aux)


class DetectionMetadataExtractor:
    DEFECT_SEVERITY  = {"cup": 0, "dustparticle": 2, "eyelash": 3, "foam": 1,
                         "hair": 3, "insect": 5, "overfill": 1, "plasticparticle": 4,
                         "residue": 2, "underfill": 1, "waterbubble": 1, "waterlayer": 2}
    DEFECT_CATEGORY  = {"cup": 0, "dustparticle": 1, "eyelash": 1, "hair": 1,
                         "insect": 1, "plasticparticle": 1, "foam": 2, "overfill": 3,
                         "underfill": 3, "residue": 4, "waterbubble": 4, "waterlayer": 4}

    def __init__(self, feature_dim=64):
        self.feature_dim = feature_dim

    def extract(self, detections, image_shape):
        h, w = image_shape[:2]
        features = []
        primary_det = max(
            (d for d in detections if d["class_name"] != "cup"),
            key=lambda x: x["conf"], default=None
        ) or (detections[0] if detections else None)

        if primary_det and "box" in primary_det:
            x1, y1, x2, y2 = primary_det["box"]
            cx, cy = (x1+x2)/2/w, (y1+y2)/2/h
            bw, bh = (x2-x1)/w, (y2-y1)/h
            area   = bw * bh
            ar     = bw / (bh + 1e-6)
            features.extend([cx, cy, bw, bh, area, min(ar, 5)/5, primary_det["conf"]])
            spatial = [0]*9
            spatial[min(2,int(cy*3))*3 + min(2,int(cx*3))] = 1
            features.extend(spatial)
            size_cat = [int(area<0.05), int(0.05<=area<0.2), int(area>=0.2)]
            features.extend(size_cat)
            shape_cat = [int(0.7<ar<1.3), int(ar>=1.3), int(ar<=0.7)]
            features.extend(shape_cat)
        else:
            features.extend([0]*22)

        class_confs  = [max((d["conf"] for d in detections if d["class_id"]==i), default=0)
                        for i in range(Config.NUM_CLASSES)]
        class_counts = [min(sum(1 for d in detections if d["class_id"]==i)/3, 1)
                        for i in range(Config.NUM_CLASSES)]
        features.extend(class_confs)
        features.extend(class_counts)

        severity = max((self.DEFECT_SEVERITY.get(d["class_name"], 0) for d in detections), default=0)
        features.append(severity / 5)
        cats = [0]*5
        for d in detections:
            if d["class_name"] in self.DEFECT_CATEGORY:
                cats[self.DEFECT_CATEGORY[d["class_name"]]] = 1
        features.extend(cats)
        features.append(min(len(detections)/5, 1))
        features.append(1 if any(d["class_name"] != "cup" for d in detections) else 0)

        features = np.array(features, dtype=np.float32)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        return features[:self.feature_dim]


def extract_roi(image, box, crop_size=64):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((crop_size, crop_size, 3), dtype=np.float32).transpose(2,0,1)
    roi = cv2.resize(image[y1:y2, x1:x2], (crop_size, crop_size))
    return (roi.astype(np.float32) / 255.0).transpose(2, 0, 1)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SENSOR CLASSIFIER INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

def engineer_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering — must match dairyfusion_sensor_classifier_v2.py exactly."""
    df = df.copy()
    slot_map = {"morning": 0, "midday": 1, "evening": 2}
    df["slot_code"]      = df["production_slot"].map(slot_map).fillna(0).astype(int)
    df["temp_below_min"] = np.maximum(0, 42.0 - df["yoghurt_temp_c"])
    df["temp_above_max"] = np.maximum(0, df["yoghurt_temp_c"] - 45.0)
    df["temp_in_range"]  = ((df["yoghurt_temp_c"] >= 42.0) &
                             (df["yoghurt_temp_c"] <= 45.0)).astype(int)
    df["raining"]        = (df["precipitation_mm"] > 0.1).astype(int)
    df["humidity_gap"]   = df["factory_humidity_pct"] - df["weather_humidity_pct"]
    return df


class SensorClassifier:
    """Wraps the trained sensor .pkl models."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        self.available = False

        type_path = model_dir / "sensor_defect_type_model.pkl"
        sev_path  = model_dir / "sensor_defect_severity_model.pkl"

        if not type_path.exists() or not sev_path.exists():
            print(f"  [Sensor] WARNING: model files not found in {model_dir}")
            print(f"  [Sensor] Run dairyfusion_sensor_classifier_v2.py first.")
            return

        with open(type_path, "rb") as f:
            self.type_bundle = pickle.load(f)
        with open(sev_path, "rb") as f:
            self.sev_bundle = pickle.load(f)

        self.available = True
        print(f"  [Sensor] Models loaded from {model_dir}")

    def predict(self,
                yoghurt_temp_c:       float,
                factory_humidity_pct: float,
                weather_temp_c:       float,
                weather_humidity_pct: float,
                precipitation_mm:     float,
                production_slot:      str = "morning") -> dict:
        """
        Returns defect_type, type_confidence, defect_severity, severity_confidence.
        If models unavailable → returns 'unknown' with 0 confidence.
        """
        if not self.available:
            return {"defect_type": "unknown", "type_confidence": 0.0,
                    "defect_severity": "unknown", "severity_confidence": 0.0}

        sample = engineer_sensor_features(pd.DataFrame([{
            "yoghurt_temp_c":       yoghurt_temp_c,
            "factory_humidity_pct": factory_humidity_pct,
            "weather_temp_c":       weather_temp_c,
            "weather_humidity_pct": weather_humidity_pct,
            "precipitation_mm":     precipitation_mm,
            "production_slot":      production_slot,
        }]))

        feat_cols = self.type_bundle["feature_cols"]
        X = sample[feat_cols].values

        pred_type = self.type_bundle["label_encoder"].inverse_transform(
            self.type_bundle["model"].predict(X))[0]
        pred_sev  = self.sev_bundle["label_encoder"].inverse_transform(
            self.sev_bundle["model"].predict(X))[0]

        type_conf = float(self.type_bundle["model"].predict_proba(X).max() * 100)
        sev_conf  = float(self.sev_bundle["model"].predict_proba(X).max() * 100)

        if type_conf < Config.SENSOR_CONFIDENCE_THRESHOLD:
            pred_type = "uncertain"
        if sev_conf < Config.SENSOR_CONFIDENCE_THRESHOLD:
            pred_sev = "uncertain"

        return {
            "defect_type":         pred_type,
            "type_confidence":     round(type_conf, 1),
            "defect_severity":     pred_sev,
            "severity_confidence": round(sev_conf, 1),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LATE FUSION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

# Maps visual class names → sensor defect type names
VISUAL_TO_SENSOR_MAP = {
    "foam":            "foam",
    "waterlayer":      "water_layer",
    "waterbubble":     "water_droplets",
    "residue":         "residue",
    "cup":             "none",
    "hair":            None,   # sensor cannot detect physical contaminants
    "insect":          None,
    "dustparticle":    None,
    "eyelash":         None,
    "plasticparticle": None,
    "overfill":        None,
    "underfill":       None,
}

# Root cause explanations for each defect type
ROOT_CAUSES = {
    "foam":          "Yoghurt temperature exceeded filling range (>44.8°C), causing foam formation",
    "water_layer":   "Yoghurt temperature outside stable range, causing syneresis (water separation)",
    "water_droplets":"High factory humidity + low yoghurt temperature causing condensation",
    "residue":       "Hot and dry OR warm and humid conditions causing surface residue",
    "hair":          "Physical contamination — hair detected on cup surface",
    "insect":        "Physical contamination — insect detected on cup surface",
    "dustparticle":  "Physical contamination — dust particle detected on cup surface",
    "eyelash":       "Physical contamination — eyelash detected on cup surface",
    "plasticparticle": "Physical contamination — plastic particle detected on cup surface",
    "overfill":      "Fill level too high — check filling nozzle calibration",
    "underfill":     "Fill level too low — check filling nozzle calibration",
    "none":          "No defect detected",
    "cup":           "No defect detected — cup present only",
}

# Recommended actions
ACTIONS = {
    "foam":          "STOP LINE → Check yoghurt heater calibration → Reduce filling temperature",
    "water_layer":   "ALERT → Adjust temperature control → Check batch consistency",
    "water_droplets":"ALERT → Reduce factory humidity → Check ventilation system",
    "residue":       "MONITOR → Clean filling nozzles → Check ambient temperature",
    "hair":          "STOP LINE → Check hygiene of operators → Replace hair nets",
    "insect":        "STOP LINE → Pest control inspection immediately",
    "dustparticle":  "STOP LINE → Check air filtration → Clean line",
    "eyelash":       "STOP LINE → Operator hygiene check",
    "plasticparticle": "STOP LINE → Check packaging materials for damage",
    "overfill":      "ADJUST → Recalibrate filling nozzle volume",
    "underfill":     "ADJUST → Recalibrate filling nozzle volume",
    "none":          "OK — No action required",
    "cup":           "OK — No action required",
}

PHYSICAL_CONTAMINANTS = {"hair", "insect", "dustparticle", "eyelash", "plasticparticle"}
ENVIRONMENTAL_DEFECTS = {"foam", "waterlayer", "waterbubble", "residue"}
FILL_ISSUES           = {"overfill", "underfill"}


def late_fusion(visual_result: dict, sensor_result: dict) -> dict:
    """
    Combines visual and sensor stream outputs into one final decision.

    Args:
        visual_result: output from visual pipeline
            {prediction_name, confidence, detections}
        sensor_result: output from sensor classifier
            {defect_type, type_confidence, defect_severity, severity_confidence}

    Returns:
        final_decision dict with all fields needed for display/logging
    """
    vis_class  = visual_result["prediction_name"]
    vis_conf   = visual_result["confidence"] * 100          # convert to %
    sen_type   = sensor_result["defect_type"]
    sen_conf   = sensor_result["type_confidence"]
    sen_sev    = sensor_result["defect_severity"]
    sen_sevcon = sensor_result["severity_confidence"]

    sensor_equiv = VISUAL_TO_SENSOR_MAP.get(vis_class, None)

    # ── CASE 1: Physical contaminant — trust visual 100% ─────────────────
    if vis_class in PHYSICAL_CONTAMINANTS:
        fusion_mode   = "Visual only — physical contaminant (sensor irrelevant)"
        final_defect  = vis_class
        final_conf    = vis_conf
        # Severity from visual confidence
        if vis_conf >= 85:   final_severity = "high"
        elif vis_conf >= 65: final_severity = "medium"
        else:                final_severity = "low"
        sensor_used   = False

    # ── CASE 2: Environmental defect — weighted fusion ────────────────────
    elif vis_class in ENVIRONMENTAL_DEFECTS:
        sensor_used = True
        # Check agreement between streams
        agrees = (sensor_equiv is not None and sen_type == sensor_equiv and
                  sen_conf >= Config.SENSOR_CONFIDENCE_THRESHOLD)

        if agrees:
            # Both agree → boost combined confidence
            combined_conf = 0.60 * vis_conf + 0.40 * sen_conf + 5.0
            fusion_mode   = "Weighted fusion — visual + sensor AGREE (+5% bonus)"
            final_defect  = vis_class
        else:
            # Disagree → visual wins on type but sensor informs severity
            combined_conf = 0.75 * vis_conf + 0.25 * sen_conf
            fusion_mode   = f"Weighted fusion — streams DISAGREE (visual wins, sensor={sen_type})"
            final_defect  = vis_class

        final_conf     = round(min(combined_conf, 99.9), 1)
        final_severity = sen_sev if sen_sev not in ("uncertain","unknown") else (
            "high" if vis_conf >= 85 else "medium" if vis_conf >= 65 else "low"
        )

    # ── CASE 3: Visual says no defect but sensor detects risk ─────────────
    elif vis_class in ("cup", "none") and sen_type not in ("none","uncertain","unknown"):
        if sen_conf >= 80:
            # Sensor confident → override visual
            fusion_mode   = f"Sensor override — visual saw no defect, sensor detected {sen_type}"
            final_defect  = sen_type.replace("_", "")   # match visual class name format
            final_conf    = sen_conf * 0.80              # slight penalty for override
            final_severity= sen_sev if sen_sev not in ("uncertain","unknown") else "low"
            sensor_used   = True
        else:
            # Sensor not confident enough to override clean visual
            fusion_mode   = "Visual wins — sensor risk below override threshold"
            final_defect  = "none"
            final_conf    = vis_conf
            final_severity= "none"
            sensor_used   = False

    # ── CASE 4: Fill level or unmatched class ─────────────────────────────
    else:
        fusion_mode   = "Visual only — fill level / no sensor equivalent"
        final_defect  = vis_class
        final_conf    = vis_conf
        final_severity= sen_sev if sen_sev not in ("uncertain","unknown") else (
            "medium" if vis_conf >= 70 else "low"
        )
        sensor_used   = True if sen_sev not in ("uncertain","unknown") else False

    root_cause = ROOT_CAUSES.get(final_defect, ROOT_CAUSES.get(vis_class, "Unknown"))
    action     = ACTIONS.get(final_defect, ACTIONS.get(vis_class, "Manual inspection required"))

    is_defect  = final_defect not in ("none", "cup")

    return {
        # Core outputs
        "defect_detected":  is_defect,
        "final_defect":     final_defect,
        "final_confidence": round(float(final_conf), 1),
        "final_severity":   final_severity,
        "root_cause":       root_cause,
        "action":           action,
        # Stream details
        "visual_class":     vis_class,
        "visual_confidence":round(vis_conf, 1),
        "sensor_type":      sen_type,
        "sensor_confidence":sen_conf,
        "sensor_severity":  sen_sev,
        "sensor_used":      sensor_used,
        "fusion_mode":      fusion_mode,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

class DairyFusionPipeline:
    """
    End-to-end DairyFusion AI pipeline.
    Loads all three components and runs them together.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        print(f"\n{'='*60}")
        print(f"  DairyFusion AI — Unified Inference Pipeline v2")
        print(f"{'='*60}")
        print(f"  Device: {config.DEVICE}")

        # ── Load visual models ─────────────────────────────────────────────
        print(f"\n[1/3] Loading YOLO model...")
        self.yolo = YOLO(config.YOLO_MODEL_PATH)

        print(f"[2/3] Loading Multimodal CNN...")
        self.multimodal = DairyNetMultimodal(config)
        self.multimodal.load_state_dict(
            torch.load(config.MULTIMODAL_MODEL_PATH, map_location=self.device)
        )
        self.multimodal.to(self.device).eval()
        self.metadata_extractor = DetectionMetadataExtractor(config.METADATA_FEATURE_DIM)

        # ── Load sensor classifier ─────────────────────────────────────────
        print(f"[3/3] Loading Sensor Classifier...")
        self.sensor = SensorClassifier(config.SENSOR_MODEL_DIR)

        print(f"\n  All components loaded. Ready.\n{'='*60}\n")

    # ── Visual stream ──────────────────────────────────────────────────────

    def run_visual(self, image_path: str) -> dict:
        """Run YOLO + Multimodal on an image. Returns visual result dict."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        img_h, img_w = image.shape[:2]
        image_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_img   = image.copy()

        # YOLO detection
        yolo_results = self.yolo.predict(image, conf=0.05, verbose=False)
        detections   = []

        if yolo_results and yolo_results[0].boxes is not None:
            for i, box in enumerate(yolo_results[0].boxes):
                cls_id   = int(box.cls[0].item())
                conf     = float(box.conf[0].item())
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                cls_name = Config.CLASS_NAMES[cls_id]

                threshold = self.config.OPTIMAL_THRESHOLDS.get(cls_name, 0.25)
                if conf < threshold:
                    continue

                x1c,y1c,x2c,y2c = correct_bbox(x1,y1,x2,y2,cls_id,img_w,img_h)
                detections.append({
                    "class_id": cls_id, "class_name": cls_name,
                    "conf": conf, "box": [x1c,y1c,x2c,y2c],
                })

        # Select primary detection for ROI
        primary = (max((d for d in detections if d["class_name"]!="cup"),
                       key=lambda x: x["conf"], default=None)
                   or (detections[0] if detections else None))

        if primary:
            roi = extract_roi(image_rgb, primary["box"], self.config.CROP_SIZE)
        else:
            cx, cy = img_w//2, img_h//2
            s = min(img_h, img_w)//2
            roi = extract_roi(image_rgb, [cx-s, cy-s, cx+s, cy+s], self.config.CROP_SIZE)

        metadata = self.metadata_extractor.extract(detections, image.shape)

        # Multimodal classification
        roi_t  = torch.tensor(roi,      dtype=torch.float32).unsqueeze(0).to(self.device)
        meta_t = torch.tensor(metadata, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.multimodal(roi_t, meta_t)
            probs  = torch.softmax(logits, dim=1)
            pred   = logits.argmax(1).item()
            conf   = probs[0, pred].item()

        # Annotate output image
        for det in detections:
            color = Config.COLORS.get(det["class_name"], (128,128,128))
            x1,y1,x2,y2 = det["box"]
            cv2.rectangle(output_img, (x1,y1), (x2,y2), color, 2)
            label = f"{det['class_name']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_img, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(output_img, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return {
            "image":           output_img,
            "prediction":      pred,
            "prediction_name": Config.CLASS_NAMES[pred],
            "confidence":      conf,
            "detections":      detections,
        }

    # ── Sensor stream ──────────────────────────────────────────────────────

    def run_sensor(self, sensor_data: dict) -> dict:
        """Run sensor classifier with provided readings."""
        return self.sensor.predict(**sensor_data)

    # ── Full fusion pipeline ───────────────────────────────────────────────

    def run(self, image_path: str, sensor_data: dict = None,
            output_path: str = None) -> dict:
        """
        Run full DairyFusion pipeline.

        Args:
            image_path:  path to yoghurt cup image
            sensor_data: dict with sensor readings (optional, uses defaults if None)
            output_path: save annotated output image here (optional)

        Returns:
            Complete fusion result dict
        """
        print(f"\n{'─'*60}")
        print(f"  Processing: {Path(image_path).name}")
        print(f"{'─'*60}")

        # ── Stream 1: Visual ───────────────────────────────────────────────
        print(f"  [Visual] Running YOLO + Multimodal...")
        visual_result = self.run_visual(image_path)
        print(f"  [Visual] → {visual_result['prediction_name']}  "
              f"({visual_result['confidence']*100:.1f}%)")

        # ── Stream 2: Sensor ───────────────────────────────────────────────
        if sensor_data is None:
            sensor_data = self.config.DEFAULT_SENSOR
            print(f"  [Sensor] Using default values (no live sensor)")
        else:
            print(f"  [Sensor] Using provided readings")

        print(f"           yoghurt_temp={sensor_data.get('yoghurt_temp_c','?')}°C  "
              f"factory_humidity={sensor_data.get('factory_humidity_pct','?')}%")

        sensor_result = self.run_sensor(sensor_data)
        print(f"  [Sensor] → {sensor_result['defect_type']}  "
              f"({sensor_result['type_confidence']}%)  "
              f"severity: {sensor_result['defect_severity']}")

        # ── Late Fusion ────────────────────────────────────────────────────
        print(f"  [Fusion] Combining streams...")
        final = late_fusion(visual_result, sensor_result)

        # ── Annotate output image with fusion result ───────────────────────
        out_img = visual_result["image"].copy()
        img_h, img_w = out_img.shape[:2]

        # Banner colour: green=ok, orange=low, red=defect
        if not final["defect_detected"]:
            banner_col = (0, 140, 0)
        elif final["final_severity"] in ("high", "medium"):
            banner_col = (0, 0, 200)
        else:
            banner_col = (0, 140, 200)

        cv2.rectangle(out_img, (0, 0), (img_w, 80), banner_col, -1)
        status = "DEFECT" if final["defect_detected"] else "CLEAN"
        cv2.putText(out_img, f"[{status}] {final['final_defect'].upper()}  "
                    f"{final['final_confidence']:.1f}%  |  severity: {final['final_severity']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(out_img, final["root_cause"][:80],
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1)
        cv2.putText(out_img, f"Action: {final['action'][:70]}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,255,200), 1)

        if output_path:
            cv2.imwrite(output_path, out_img)
            print(f"  [Output] Saved → {output_path}")

        # ── Print final result ─────────────────────────────────────────────
        print(f"\n  ┌─ FINAL FUSION RESULT {'─'*38}")
        print(f"  │  Defect       : {final['final_defect']}")
        print(f"  │  Confidence   : {final['final_confidence']}%")
        print(f"  │  Severity     : {final['final_severity']}")
        print(f"  │  Root cause   : {final['root_cause']}")
        print(f"  │  Action       : {final['action']}")
        print(f"  │  Fusion mode  : {final['fusion_mode']}")
        print(f"  └{'─'*60}")

        final["annotated_image"] = out_img
        return final


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DairyFusion AI — Unified Inference")
    p.add_argument("--image",            required=True, help="Path to cup image")
    p.add_argument("--output",           default="dairyfusion_output.jpg",
                                         help="Output image path")
    p.add_argument("--sensor-model-dir", default="sensor_classifier_output",
                                         help="Folder with .pkl sensor models")
    p.add_argument("--yolo-model",       default="dairynet_output_yolo_v7/train/weights/best.pt")
    p.add_argument("--multimodal-model", default="dairynet_multimodal_v4/best_multimodal.pt")
    p.add_argument("--slot",             default="morning",
                                         choices=["morning","midday","evening"])
    # Sensor values (if not using live ESP32)
    p.add_argument("--yoghurt-temp",       type=float, default=None)
    p.add_argument("--factory-humidity",   type=float, default=None)
    p.add_argument("--weather-temp",       type=float, default=None)
    p.add_argument("--weather-humidity",   type=float, default=None)
    p.add_argument("--precipitation",      type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    config = Config()
    config.YOLO_MODEL_PATH       = args.yolo_model
    config.MULTIMODAL_MODEL_PATH = args.multimodal_model
    config.SENSOR_MODEL_DIR      = args.sensor_model_dir

    pipeline = DairyFusionPipeline(config)

    # Build sensor data dict
    if any(v is not None for v in [args.yoghurt_temp, args.factory_humidity,
                                    args.weather_temp, args.weather_humidity,
                                    args.precipitation]):
        sensor_data = {
            "yoghurt_temp_c":       args.yoghurt_temp       or Config.DEFAULT_SENSOR["yoghurt_temp_c"],
            "factory_humidity_pct": args.factory_humidity   or Config.DEFAULT_SENSOR["factory_humidity_pct"],
            "weather_temp_c":       args.weather_temp       or Config.DEFAULT_SENSOR["weather_temp_c"],
            "weather_humidity_pct": args.weather_humidity   or Config.DEFAULT_SENSOR["weather_humidity_pct"],
            "precipitation_mm":     args.precipitation      or 0.0,
            "production_slot":      args.slot,
        }
    else:
        sensor_data = None   # will use defaults

    result = pipeline.run(
        image_path  = args.image,
        sensor_data = sensor_data,
        output_path = args.output,
    )

    # Save JSON result
    json_path = Path(args.output).with_suffix(".json")
    result_to_save = {k: v for k, v in result.items() if k != "annotated_image"}
    with open(json_path, "w") as f:
        json.dump(result_to_save, f, indent=2)
    print(f"\n  [JSON]   Saved → {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PROGRAMMATIC API — use this in your UI or ESP32 integration
# ─────────────────────────────────────────────────────────────────────────────

def create_pipeline(
    sensor_model_dir:      str = "sensor_classifier_output",
    yolo_model_path:       str = "dairynet_output_yolo_v7/train/weights/best.pt",
    multimodal_model_path: str = "dairynet_multimodal_v4/best_multimodal.pt",
) -> DairyFusionPipeline:
    """
    Create and return a DairyFusionPipeline instance.
    Use this when integrating into a UI or calling from another script.

    Example:
        from final_inference_v2 import create_pipeline

        pipeline = create_pipeline()
        result = pipeline.run(
            image_path  = "cup_017.jpg",
            sensor_data = {
                "yoghurt_temp_c":       41.5,
                "factory_humidity_pct": 85.0,
                "weather_temp_c":       27.0,
                "weather_humidity_pct": 88.0,
                "precipitation_mm":     2.5,
                "production_slot":      "evening",
            },
            output_path = "result_017.jpg"
        )
        print(result["final_defect"], result["root_cause"], result["action"])
    """
    config = Config()
    config.SENSOR_MODEL_DIR      = sensor_model_dir
    config.YOLO_MODEL_PATH       = yolo_model_path
    config.MULTIMODAL_MODEL_PATH = multimodal_model_path
    return DairyFusionPipeline(config)


if __name__ == "__main__":
    main()
