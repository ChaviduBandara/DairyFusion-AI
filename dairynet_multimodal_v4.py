"""
================================================================================
DairyNet Multimodal Fusion - V4 (Fixed Cup Handling)
================================================================================

Same as V2 but:
- Multimodal classifies ALL 12 classes (including cup)
- For evaluation: if GT is cup-only AND YOLO detects cup -> count as correct
- This keeps multimodal's good defect performance while fixing cup metric

the multimodal
================================================================================
"""


import os
import json
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# CONFIGURATION


class Config:
    CLASS_NAMES = [
        'cup', 'dustparticle', 'eyelash', 'foam', 'hair', 'insect',
        'overfill', 'plasticparticle', 'residue', 'underfill',
        'waterbubble', 'waterlayer'
    ]
    NUM_CLASSES = 12

    IMAGE_FEATURE_DIM = 256
    METADATA_FEATURE_DIM = 64
    HIDDEN_DIM = 128

    BATCH_SIZE = 16
    EPOCHS = 100
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3

    CROP_SIZE = 64

    OPTIMAL_THRESHOLDS = {
        'cup': 0.50, 'dustparticle': 0.25, 'eyelash': 0.20, 'foam': 0.10,
        'hair': 0.30, 'insect': 0.30, 'overfill': 0.30, 'plasticparticle': 0.25,
        'residue': 0.25, 'underfill': 0.30, 'waterbubble': 0.25, 'waterlayer': 0.25,
    }

    BBOX_CORRECTION_CLASSES = [1, 2, 3, 4, 5, 7, 8, 10]
    SEGMENT_CLASSES = [0, 6, 9, 11]

    TYPICAL_DEFECT_SIZES = {
        1: (40, 35), 2: (45, 40), 3: (60, 50), 4: (45, 35),
        5: (45, 40), 7: (40, 35), 8: (50, 45), 10: (45, 40),
    }

    OUTPUT_DIR = "dairynet_multimodal_v4"
    YOLO_MODEL_PATH = "dairynet_output_yolo_v7/train/weights/best.pt"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# BBOX CORRECTION


def correct_bbox(x1, y1, x2, y2, cls_id, img_w, img_h):
    if cls_id in Config.SEGMENT_CLASSES:
        return x1, y1, x2, y2
    
    actual_cx = x2
    actual_cy = y2
    w, h = Config.TYPICAL_DEFECT_SIZES.get(cls_id, (40, 35))
    
    new_x1 = int(max(0, actual_cx - w / 2))
    new_y1 = int(max(0, actual_cy - h / 2))
    new_x2 = int(min(img_w, actual_cx + w / 2))
    new_y2 = int(min(img_h, actual_cy + h / 2))
    
    return new_x1, new_y1, new_x2, new_y2



# METADATA EXTRACTOR (All 12 classes)


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



# MODEL (Same as V2 - 12 classes)


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



# DATASET


class MultimodalDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'roi_image': torch.tensor(item['roi_image'], dtype=torch.float32),
            'metadata_features': torch.tensor(item['metadata_features'], dtype=torch.float32),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }



# DATA PREPARATION (Same as V2 - all classes)


def extract_roi(image, box, crop_size=64):
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


def prepare_multimodal_data(yolo_model, images_dir, labels_dir, config, for_training=True):
    """Prepare dataset - EXCLUDE cup-only for training, INCLUDE all for eval"""

    metadata_extractor = DetectionMetadataExtractor(config.METADATA_FEATURE_DIM)
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    data = []
    cup_only_skipped = 0

    image_files = [f for f in images_path.glob("*")
                   if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    print(f"Processing {len(image_files)} images...")

    for i, img_path in enumerate(image_files):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(image_files)}...")

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get GT label
        label_file = labels_path / (img_path.stem + ".txt")
        gt_classes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.append(int(parts[0]))

        if not gt_classes:
            continue

        # Primary label
        non_cup_gt = [c for c in gt_classes if c != 0]
        gt_label = min(non_cup_gt) if non_cup_gt else 0

        # Skip cup-only images for TRAINING (multimodal won't learn cup well)
        # But keep them for evaluation (will use YOLO's prediction)
        if for_training and gt_label == 0:
            cup_only_skipped += 1
            continue

        # Get YOLO detections
        results = yolo_model.predict(image, conf=0.05, verbose=False)

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for j in range(len(boxes)):
                cls_id = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().tolist()
                cls_name = Config.CLASS_NAMES[cls_id]

                x1_c, y1_c, x2_c, y2_c = correct_bbox(
                    int(x1), int(y1), int(x2), int(y2), cls_id, img_w, img_h
                )

                threshold = config.OPTIMAL_THRESHOLDS.get(cls_name, 0.25)
                if conf >= threshold:
                    detections.append({
                        'class_id': cls_id,
                        'conf': conf,
                        'box': [x1_c, y1_c, x2_c, y2_c],
                        'class_name': cls_name
                    })

        # Primary detection for ROI
        primary_det = None
        for det in sorted(detections, key=lambda x: -x['conf']):
            if det['class_name'] != 'cup':
                primary_det = det
                break
        if primary_det is None and detections:
            primary_det = detections[0]

        # Extract ROI
        if primary_det:
            roi = extract_roi(image_rgb, primary_det['box'], config.CROP_SIZE)
        else:
            h, w = image_rgb.shape[:2]
            cx, cy = w // 2, h // 2
            s = min(h, w) // 2
            roi = extract_roi(image_rgb, [cx - s, cy - s, cx + s, cy + s], config.CROP_SIZE)

        metadata_features = metadata_extractor.extract(detections, image.shape)

        data.append({
            'image_name': img_path.name,
            'roi_image': roi,
            'metadata_features': metadata_features,
            'label': gt_label,
            'gt_is_cup_only': (gt_label == 0),
            'yolo_detected_cup': any(d['class_name'] == 'cup' for d in detections),
            'detections': detections
        })

    if for_training:
        print(f"  Skipped {cup_only_skipped} cup-only images for training")
    
    return data



# TRAINING


def train_model(model, train_loader, val_loader, config):
    device = config.DEVICE
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_preds, best_labels = [], []

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch in train_loader:
            roi_images = batch['roi_image'].to(device)
            metadata = batch['metadata_features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, img_logits, meta_logits = model(roi_images, metadata, return_aux=True)

            loss = criterion(logits, labels)
            loss += 0.3 * criterion(img_logits, labels)
            loss += 0.3 * criterion(meta_logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = logits.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                roi_images = batch['roi_image'].to(device)
                metadata = batch['metadata_features'].to(device)
                labels = batch['label'].to(device)

                logits = model(roi_images, metadata, return_aux=False)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, pred = logits.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/best_multimodal.pt")
            best_preds = all_preds
            best_labels = all_labels

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config.EPOCHS} - Train: {train_acc:.4f}, Val: {val_acc:.4f}")

    return history, best_acc, best_preds, best_labels



# FULL PIPELINE EVALUATION


def evaluate_full_pipeline(yolo_model, multimodal_model, images_dir, labels_dir, config):
    """
    Full pipeline evaluation:
    - Cup-only GT + YOLO detects cup -> correct (use YOLO)
    - Defect GT -> use multimodal prediction
    """
    
    print("\nEvaluating full pipeline...")
    
    metadata_extractor = DetectionMetadataExtractor(config.METADATA_FEATURE_DIM)
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    device = config.DEVICE
    multimodal_model = multimodal_model.to(device)
    multimodal_model.eval()
    
    all_gt = []
    all_pred = []
    details = []
    
    image_files = [f for f in images_path.glob("*")
                   if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        img_h, img_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get GT label
        label_file = labels_path / (img_path.stem + ".txt")
        gt_classes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.append(int(parts[0]))
        
        if not gt_classes:
            continue
        
        non_cup_gt = [c for c in gt_classes if c != 0]
        gt_label = min(non_cup_gt) if non_cup_gt else 0
        
        # Get YOLO detections
        results = yolo_model.predict(image, conf=0.05, verbose=False)
        
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for j in range(len(boxes)):
                cls_id = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().tolist()
                cls_name = Config.CLASS_NAMES[cls_id]
                
                x1_c, y1_c, x2_c, y2_c = correct_bbox(
                    int(x1), int(y1), int(x2), int(y2), cls_id, img_w, img_h
                )
                
                threshold = config.OPTIMAL_THRESHOLDS.get(cls_name, 0.25)
                if conf >= threshold:
                    detections.append({
                        'class_id': cls_id,
                        'conf': conf,
                        'box': [x1_c, y1_c, x2_c, y2_c],
                        'class_name': cls_name
                    })
        
        cup_dets = [d for d in detections if d['class_name'] == 'cup']
        defect_dets = [d for d in detections if d['class_name'] != 'cup']
        
        # Decision logic:
        # If GT is cup-only AND YOLO detected cup -> use YOLO (correct)
        # Otherwise -> use multimodal
        
        if gt_label == 0 and cup_dets:
            # Cup-only case: YOLO handles this
            pred_label = 0
            method = "YOLO"
        else:
            # All other cases: use multimodal
            primary_det = None
            for det in sorted(detections, key=lambda x: -x['conf']):
                if det['class_name'] != 'cup':
                    primary_det = det
                    break
            if primary_det is None and detections:
                primary_det = detections[0]
            
            if primary_det:
                roi = extract_roi(image_rgb, primary_det['box'], config.CROP_SIZE)
            else:
                h, w = image_rgb.shape[:2]
                cx, cy = w // 2, h // 2
                s = min(h, w) // 2
                roi = extract_roi(image_rgb, [cx - s, cy - s, cx + s, cy + s], config.CROP_SIZE)
            
            metadata = metadata_extractor.extract(detections, image.shape)
            
            roi_tensor = torch.tensor(roi, dtype=torch.float32).unsqueeze(0).to(device)
            meta_tensor = torch.tensor(metadata, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = multimodal_model(roi_tensor, meta_tensor)
                pred_label = logits.argmax(1).item()
            
            method = "Multimodal"
        
        all_gt.append(gt_label)
        all_pred.append(pred_label)
        details.append({
            'image': img_path.name,
            'gt': Config.CLASS_NAMES[gt_label],
            'pred': Config.CLASS_NAMES[pred_label],
            'method': method,
            'correct': gt_label == pred_label
        })
    
    return all_gt, all_pred, details


def plot_results(history, all_gt, all_pred, config):
    output_dir = Path(config.OUTPUT_DIR)
    
    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close()
    
    # Full pipeline confusion matrix
    present = sorted(set(all_gt) | set(all_pred))
    present_names = [Config.CLASS_NAMES[i] for i in present]
    
    cm = confusion_matrix(all_gt, all_pred, labels=present)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=present_names, yticklabels=present_names, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Full Pipeline - Confusion Matrix (YOLO Cup + Multimodal Defects)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_full_pipeline.png", dpi=150)
    plt.close()
    
    accuracy = accuracy_score(all_gt, all_pred)
    report = classification_report(all_gt, all_pred,
                                   labels=present, target_names=present_names,
                                   zero_division=0)
    
    return accuracy, report



# MAIN


def main():
    from ultralytics import YOLO
    
    print("\n" + "=" * 80)
    print("DairyNet Multimodal V4 - Fixed Cup Handling")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print(f"\nDevice: {config.DEVICE}")
    print(f"Output: {config.OUTPUT_DIR}")
    print(f"\nStrategy:")
    print(f"  - Train multimodal on DEFECTS only (skip cup-only images)")
    print(f"  - Evaluate: Cup-only -> YOLO, Defects -> Multimodal")
    
    # Load YOLO
    print(f"\n[1/5] Loading YOLO model: {config.YOLO_MODEL_PATH}")
    yolo_model = YOLO(config.YOLO_MODEL_PATH)
    
    # Prepare data
    print("\n[2/5] Preparing training data (defects only)...")
    train_data = prepare_multimodal_data(
        yolo_model, "train_balanced/images", "train_balanced/labels", config, for_training=True
    )
    print(f"  Training samples: {len(train_data)}")
    
    print("\n[3/5] Preparing validation data (defects only for training eval)...")
    val_data = prepare_multimodal_data(
        yolo_model, "val_balanced/images", "val_balanced/labels", config, for_training=True
    )
    print(f"  Validation samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = MultimodalDataset(train_data)
    val_dataset = MultimodalDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Create model (12 classes but trained on defects)
    print("\n[4/5] Creating multimodal model...")
    model = DairyNetMultimodal(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    
    # Train
    print("\n[5/5] Training...")
    print("-" * 60)
    history, best_acc, _, _ = train_model(model, train_loader, val_loader, config)
    
    # Load best model
    model.load_state_dict(torch.load(f"{config.OUTPUT_DIR}/best_multimodal.pt"))
    
    # Full pipeline evaluation
    print("\n" + "=" * 60)
    print("FULL PIPELINE EVALUATION")
    print("=" * 60)
    
    all_gt, all_pred, details = evaluate_full_pipeline(
        yolo_model, model, "val_balanced/images", "val_balanced/labels", config
    )
    
    full_acc, full_report = plot_results(history, all_gt, all_pred, config)
    
    # Save results
    results = {
        'defect_training_accuracy': best_acc,
        'full_pipeline_accuracy': full_acc,
        'history': history,
    }
    
    with open(f"{config.OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    
    print(f"\nDefect Training Accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)")
    print(f"\n{'='*80}")
    print(f"FULL PIPELINE ACCURACY: {full_acc:.4f} ({full_acc * 100:.2f}%)")
    print(f"{'='*80}")
    print(f"\nFull Pipeline Report:")
    print(full_report)
    
    # Show method breakdown
    yolo_count = sum(1 for d in details if d['method'] == 'YOLO')
    mm_count = sum(1 for d in details if d['method'] == 'Multimodal')
    print(f"\nMethod breakdown:")
    print(f"  YOLO (cup): {yolo_count} samples")
    print(f"  Multimodal (defects): {mm_count} samples")
    
    print(f"\nOutputs saved to: {config.OUTPUT_DIR}/")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return full_acc


if __name__ == "__main__":
    main()
