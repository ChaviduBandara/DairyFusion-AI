
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn

from ultralytics import YOLO


# CONFIGURATION

class Config:
    CLASS_NAMES = [
        'cup', 'dustparticle', 'eyelash', 'foam', 'hair', 'insect',
        'overfill', 'plasticparticle', 'residue', 'underfill',
        'waterbubble', 'waterlayer'
    ]
    NUM_CLASSES = 12

    SINGLE_IMAGE_PATH = "IMG_3094_jpg.rf.25c19746c5563bd995ca246aa6ba3f7e.jpg"
    SINGLE_OUTPUT_PATH = "single_inference_output2.jpg"

    IMAGE_FEATURE_DIM = 256
    METADATA_FEATURE_DIM = 64
    HIDDEN_DIM = 128
    DROPOUT = 0.3
    CROP_SIZE = 64

    OPTIMAL_THRESHOLDS = {
        'cup': 0.50, 'dustparticle': 0.25, 'eyelash': 0.20, 'foam': 0.10,
        'hair': 0.30, 'insect': 0.30, 'overfill': 0.30, 'plasticparticle': 0.25,
        'residue': 0.25, 'underfill': 0.30, 'waterbubble': 0.25, 'waterlayer': 0.25,
    }

    # Classes needing bbox correction (bottom-right = center)
    BBOX_CORRECTION_CLASSES = [1, 2, 3, 5, 7, 8, 10]  # Removed hair (4)
    SEGMENT_CLASSES = [0, 6, 9, 11]

    TYPICAL_DEFECT_SIZES = {
        1: (40, 35),  # dustparticle
        2: (45, 40),  # eyelash
        3: (60, 50),  # foam
        5: (45, 40),  # insect
        7: (40, 35),  # plasticparticle
        8: (50, 45),  # residue
        10: (45, 40),  # waterbubble
    }

    # Paths
    YOLO_MODEL_PATH = "dairynet_output_yolo_v7/train/weights/best.pt"
    MULTIMODAL_MODEL_PATH = "dairynet_multimodal_v4/best_multimodal.pt"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Colors for visualization (BGR)
    COLORS = {
        'cup': (255, 200, 0),
        'dustparticle': (0, 0, 255),
        'eyelash': (0, 255, 0),
        'foam': (255, 0, 0),
        'hair': (255, 0, 255),
        'insect': (255, 255, 0),
        'overfill': (0, 255, 128),
        'plasticparticle': (0, 128, 255),
        'residue': (128, 0, 255),
        'underfill': (255, 128, 0),
        'waterbubble': (0, 255, 255),
        'waterlayer': (128, 255, 0),
    }

# BBOX CORRECTION

def correct_bbox(x1, y1, x2, y2, cls_id, img_w, img_h):
    """Correct bbox for classes with wrong format"""

    # Segment classes work correctly
    if cls_id in Config.SEGMENT_CLASSES:
        return x1, y1, x2, y2

    # Hair (class 4) - no correction needed, original detection is good
    if cls_id == 4:
        return x1, y1, x2, y2

    # For other bbox classes: bottom-right = actual center
    if cls_id in Config.BBOX_CORRECTION_CLASSES:
        actual_cx = x2
        actual_cy = y2
        w, h = Config.TYPICAL_DEFECT_SIZES.get(cls_id, (40, 35))

        new_x1 = int(max(0, actual_cx - w / 2))
        new_y1 = int(max(0, actual_cy - h / 2))
        new_x2 = int(min(img_w, actual_cx + w / 2))
        new_y2 = int(min(img_h, actual_cy + h / 2))

        return new_x1, new_y1, new_x2, new_y2

    return x1, y1, x2, y2

# MODEL DEFINITIONS (must match training)

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
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class MultimodalFusionClassifier(nn.Module):
    def __init__(self, image_dim=256, metadata_dim=64, hidden_dim=128,
                 num_classes=12, dropout=0.3):
        super().__init__()

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.image_aux = nn.Linear(hidden_dim, num_classes)
        self.metadata_aux = nn.Linear(hidden_dim, num_classes)

    def forward(self, image_features, metadata_features, return_aux=False):
        img_proj = self.image_proj(image_features)
        meta_proj = self.metadata_proj(metadata_features)

        img_seq = img_proj.unsqueeze(1)
        meta_seq = meta_proj.unsqueeze(1)
        attended, _ = self.cross_attention(img_seq, meta_seq, meta_seq)
        attended = attended.squeeze(1)

        concat = torch.cat([img_proj, meta_proj], dim=-1)
        gate = self.gate(concat)
        gated_img = img_proj * gate
        gated_meta = meta_proj * (1 - gate)

        fused = torch.cat([gated_img + attended, gated_meta], dim=-1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)

        if return_aux:
            return logits, self.image_aux(img_proj), self.metadata_aux(meta_proj)
        return logits


class DairyNetMultimodal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roi_extractor = ROIFeatureExtractor(config.IMAGE_FEATURE_DIM)
        self.fusion_classifier = MultimodalFusionClassifier(
            image_dim=config.IMAGE_FEATURE_DIM,
            metadata_dim=config.METADATA_FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )

    def forward(self, roi_images, metadata_features, return_aux=False):
        img_features = self.roi_extractor(roi_images)
        return self.fusion_classifier(img_features, metadata_features, return_aux)



# METADATA EXTRACTOR


class DetectionMetadataExtractor:
    DEFECT_SEVERITY = {
        'cup': 0, 'dustparticle': 2, 'eyelash': 3, 'foam': 1,
        'hair': 3, 'insect': 5, 'overfill': 1, 'plasticparticle': 4,
        'residue': 2, 'underfill': 1, 'waterbubble': 1, 'waterlayer': 2
    }

    DEFECT_CATEGORY = {
        'cup': 0,
        'dustparticle': 1, 'eyelash': 1, 'hair': 1, 'insect': 1, 'plasticparticle': 1,
        'foam': 2,
        'overfill': 3, 'underfill': 3,
        'residue': 4, 'waterbubble': 4, 'waterlayer': 4,
    }

    def __init__(self, feature_dim=64):
        self.feature_dim = feature_dim

    def extract(self, detections, image_shape):
        h, w = image_shape[:2]
        features = []

        # Primary detection (non-cup preferred)
        primary_det = None
        max_conf = 0
        for det in detections:
            if det['class_name'] != 'cup' and det['conf'] > max_conf:
                max_conf = det['conf']
                primary_det = det

        if primary_det is None and detections:
            primary_det = max(detections, key=lambda x: x['conf'])

        if primary_det and 'box' in primary_det:
            x1, y1, x2, y2 = primary_det['box']

            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            area = bw * bh
            aspect_ratio = bw / (bh + 1e-6)
            features.extend([cx, cy, bw, bh, area, min(aspect_ratio, 5) / 5])

            features.append(primary_det['conf'])

            spatial = [0] * 9
            grid_x = min(2, int(cx * 3))
            grid_y = min(2, int(cy * 3))
            spatial[grid_y * 3 + grid_x] = 1
            features.extend(spatial)

            size_cat = [0, 0, 0]
            if area < 0.05:
                size_cat[0] = 1
            elif area < 0.2:
                size_cat[1] = 1
            else:
                size_cat[2] = 1
            features.extend(size_cat)

            shape_cat = [0, 0, 0]
            if 0.7 < aspect_ratio < 1.3:
                shape_cat[0] = 1
            elif aspect_ratio >= 1.3:
                shape_cat[1] = 1
            else:
                shape_cat[2] = 1
            features.extend(shape_cat)
        else:
            features.extend([0] * 22)

        class_confs = [0] * Config.NUM_CLASSES
        for det in detections:
            cls_id = det['class_id']
            if cls_id < Config.NUM_CLASSES:
                class_confs[cls_id] = max(class_confs[cls_id], det['conf'])
        features.extend(class_confs)

        class_counts = [0] * Config.NUM_CLASSES
        for det in detections:
            cls_id = det['class_id']
            if cls_id < Config.NUM_CLASSES:
                class_counts[cls_id] += 1
        class_counts = [min(c / 3, 1) for c in class_counts]
        features.extend(class_counts)

        severity = 0
        for det in detections:
            cls_name = det.get('class_name', '')
            if cls_name in self.DEFECT_SEVERITY:
                severity = max(severity, self.DEFECT_SEVERITY[cls_name])
        features.append(severity / 5)

        categories = [0] * 5
        for det in detections:
            cls_name = det.get('class_name', '')
            if cls_name in self.DEFECT_CATEGORY:
                categories[self.DEFECT_CATEGORY[cls_name]] = 1
        features.extend(categories)

        features.append(min(len(detections) / 5, 1))
        has_defect = any(det['class_name'] != 'cup' for det in detections)
        features.append(1 if has_defect else 0)

        features = np.array(features, dtype=np.float32)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = features[:self.feature_dim]

        return features



# UTILITY FUNCTIONS

def extract_roi(image, box, crop_size=64):
    """Extract and resize ROI from image"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((crop_size, crop_size, 3), dtype=np.float32)

    roi = image[y1:y2, x1:x2]
    roi = cv2.resize(roi, (crop_size, crop_size))
    roi = roi.astype(np.float32) / 255.0
    return roi.transpose(2, 0, 1)

def draw_detection(image, box, cls_name, conf, color, is_corrected=False):
    """Draw detection box on image"""
    x1, y1, x2, y2 = map(int, box)

    # Draw box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Label
    label = f"{cls_name} {conf:.2f}"
    if is_corrected:
        label += " (c)"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



# MAIN INFERENCE PIPELINE
class FullPipelineInference:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        # Load YOLO
        print(f"Loading YOLO model: {config.YOLO_MODEL_PATH}")
        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)

        # Load Multimodal
        print(f"Loading Multimodal model: {config.MULTIMODAL_MODEL_PATH}")
        self.multimodal_model = DairyNetMultimodal(config)
        self.multimodal_model.load_state_dict(
            torch.load(config.MULTIMODAL_MODEL_PATH, map_location=self.device)
        )
        self.multimodal_model.to(self.device)
        self.multimodal_model.eval()

        # Metadata extractor
        self.metadata_extractor = DetectionMetadataExtractor(config.METADATA_FEATURE_DIM)

    def run_single_image(self, image_path, output_path):
        print(f"\nRunning single-image inference:")
        print(f"  Image:  {image_path}")
        print(f"  Output: {output_path}")

        result = self.process_image(Path(image_path))

        if result is None:
            print("Failed to load image.")
            return

        # Save output image
        cv2.imwrite(output_path, result["image"])

        print(f"Prediction : {result['prediction_name']}")
        print(f"Confidence : {result['confidence']:.4f}")
        print(f"Method     : {result['method']}")
        print(f"Detections : {len(result['detections'])}")
        print("\nSaved annotated output.")


    def process_image(self, image_path):
        """Run full pipeline on single image"""

        image = cv2.imread(str(image_path))
        if image is None:
            return None

        img_h, img_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = image.copy()

        # Run YOLO
        results = self.yolo_model.predict(image, conf=0.05, verbose=False)

        detections = []
        raw_detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                cls_name = Config.CLASS_NAMES[cls_id]

                # Store raw detection
                raw_detections.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'conf': conf,
                    'box': [int(x1), int(y1), int(x2), int(y2)]
                })

                # Apply bbox correction
                x1_c, y1_c, x2_c, y2_c = correct_bbox(
                    int(x1), int(y1), int(x2), int(y2), cls_id, img_w, img_h
                )

                # Apply threshold
                threshold = self.config.OPTIMAL_THRESHOLDS.get(cls_name, 0.25)
                if conf >= threshold:
                    is_corrected = (x1_c != int(x1) or y1_c != int(y1))
                    detections.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'conf': conf,
                        'box': [x1_c, y1_c, x2_c, y2_c],
                        'original_box': [int(x1), int(y1), int(x2), int(y2)],
                        'is_corrected': is_corrected
                    })

        # Separate cup and defect detections
        cup_dets = [d for d in detections if d['class_name'] == 'cup']
        defect_dets = [d for d in detections if d['class_name'] != 'cup']

        # Always use multimodal for final classification
        # YOLO is only used for detection, not classification decision
        primary_det = None
        for det in sorted(detections, key=lambda x: -x['conf']):
            if det['class_name'] != 'cup':
                primary_det = det
                break
        if primary_det is None and detections:
            primary_det = detections[0]

        # Extract ROI
        if primary_det:
            roi = extract_roi(image_rgb, primary_det['box'], self.config.CROP_SIZE)
        else:
            h, w = image_rgb.shape[:2]
            cx, cy = w // 2, h // 2
            s = min(h, w) // 2
            roi = extract_roi(image_rgb, [cx - s, cy - s, cx + s, cy + s], self.config.CROP_SIZE)

        # Extract metadata
        metadata = self.metadata_extractor.extract(detections, image.shape)

        # Run multimodal
        roi_tensor = torch.tensor(roi, dtype=torch.float32).unsqueeze(0).to(self.device)
        meta_tensor = torch.tensor(metadata, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.multimodal_model(roi_tensor, meta_tensor)
            probs = torch.softmax(logits, dim=1)
            final_prediction = logits.argmax(1).item()
            confidence = probs[0, final_prediction].item()

        prediction_method = "Multimodal"
        if not defect_dets and cup_dets:
            prediction_method = "Multimodal (cup-only detection)"

        final_class_name = Config.CLASS_NAMES[final_prediction]

        # Draw detections on output image
        for det in detections:
            color = Config.COLORS.get(det['class_name'], (128, 128, 128))
            draw_detection(output, det['box'], det['class_name'], det['conf'],
                           color, det.get('is_corrected', False))

        # Add final prediction banner
        cv2.rectangle(output, (0, 0), (img_w, 60), (40, 40, 40), -1)

        pred_text = f"Prediction: {final_class_name} ({confidence:.2f}) via {prediction_method}"
        cv2.putText(output, pred_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return {
            'image': output,
            'prediction': final_prediction,
            'prediction_name': final_class_name,
            'confidence': confidence,
            'method': prediction_method,
            'detections': detections,
            'raw_detections': raw_detections
        }

# MAIN
def main():
    
    print("DairyNet Single Image Inference")
    print("YOLO + Multimodal Fusion")
    

    config = Config()
    print(f"Device: {config.DEVICE}")

    pipeline = FullPipelineInference(config)

    pipeline.run_single_image(
        config.SINGLE_IMAGE_PATH,
        config.SINGLE_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
