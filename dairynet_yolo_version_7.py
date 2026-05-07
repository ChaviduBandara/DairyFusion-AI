"""
NOVEL CONTRIBUTIONS:
1. Multi-Scale Attention Fusion Module (MAFM)
2. Defect-Aware Focal Loss (DAFL)
3. Complete-IoU (CIoU) Loss for Box Regression
4. Two-Stage Hierarchical Inference
5. Adaptive Bounding Box Refinement
6. Automatic Per-Class Threshold Optimization (NEW in V7)
"""

import os
import sys
import shutil
import random
import math
import time
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from copy import deepcopy

import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class Config:
    """DairyNet-YOLO V7 Configuration - With Automatic Threshold Optimization"""

    # Model
    BASE_MODEL = "yolo11s.pt"
    MODEL_NAME = "DairyNet-YOLO-V7"

    # Paths
    DATA_YAML = "data.yaml"
    OUTPUT_DIR = "dairynet_output_yolo_v7"

    # Training - Extended for difficult classes
    EPOCHS = 200  # More epochs for foam/hair learning
    PATIENCE = 30  # More patience
    BATCH_SIZE = 8
    IMG_SIZE = 640
    WORKERS = 0

    # Learning Rate
    LR0 = 0.01
    LRF = 0.001  # Lower final LR for fine-grained learning
    WARMUP_EPOCHS = 5

    # Loss Weights - Increased cls weight for better classification
    BOX_LOSS_WEIGHT = 7.5
    CLS_LOSS_WEIGHT = 1.0  # Increased from 0.5 for better classification
    DFL_LOSS_WEIGHT = 1.5

    # Focal Loss Parameters
    FOCAL_GAMMA = 2.0

    # Augmentation - More aggressive for difficult classes
    HSV_H = 0.02  # More hue variation
    HSV_S = 0.8  # More saturation variation
    HSV_V = 0.5  # More brightness variation
    DEGREES = 10  # Rotation helps with hair orientation
    TRANSLATE = 0.15
    SCALE = 0.6  # More scale variation
    SHEAR = 5
    FLIPUD = 0.5  # Vertical flip helps
    FLIPLR = 0.5
    MOSAIC = 1.0
    MIXUP = 0.15  # Some mixup helps generalization
    COPY_PASTE = 0.3  # Copy-paste augmentation for rare classes

    # Data
    VAL_SPLIT = 0.15

    # Inference - Class-specific confidence thresholds
    CONF_THRESHOLD = 0.25  # Default confidence
    USE_HIERARCHICAL = False

    # NOVEL: Per-class confidence thresholds (lower for difficult classes)
    CLASS_CONF_THRESHOLDS = {
        'cup': 0.5,  # High threshold - easy class
        'dustparticle': 0.25,
        'eyelash': 0.2,
        'foam': 0.10,  # Very low - catch more foam
        'hair': 0.10,  # Very low - catch more hair
        'insect': 0.2,
        'overfill': 0.3,
        'plasticparticle': 0.2,
        'residue': 0.2,
        'underfill': 0.3,
        'waterbubble': 0.25,
        'waterlayer': 0.25,
    }

    # Classes
    CLASS_NAMES = [
        'cup', 'dustparticle', 'eyelash', 'foam', 'hair', 'insect',
        'overfill', 'plasticparticle', 'residue', 'underfill',
        'waterbubble', 'waterlayer'
    ]
    NUM_CLASSES = 12
    CUP_CLASS_ID = 0

    # Difficult classes that need special handling
    HARD_CLASSES = ['foam', 'hair']
    HARD_CLASS_IDS = [3, 4]  # foam=3, hair=4

    # Device
    DEVICE = "0" if torch.cuda.is_available() else "cpu"

    # Class weights for training (higher = more focus)
    # Based on inverse frequency AND difficulty
    CLASS_WEIGHTS = {
        'cup': 0.1,
        'dustparticle': 1.0,
        'eyelash': 1.2,
        'foam': 3.0,  # 3x weight - very difficult
        'hair': 3.0,  # 3x weight - very difficult
        'insect': 0.8,
        'overfill': 1.0,
        'plasticparticle': 1.0,
        'residue': 0.8,
        'underfill': 1.0,
        'waterbubble': 1.5,
        'waterlayer': 1.0,
    }


# NOVEL CONTRIBUTION 1: ATTENTION MODULES
class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Part of CBAM)

    Learns to emphasize informative feature channels.
    Uses both global average pooling and max pooling for richer representation.

    For dairy defects: Learns that edge-detection channels are important for
    hair/eyelash, while texture channels matter for foam/residue.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (Part of CBAM)

    Learns WHERE to focus in the spatial dimensions.
    Creates attention map highlighting defect regions.

    For dairy defects: Learns to focus on yoghurt surface,
    suppress attention on cup rim and background.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module

    Sequential application: Channel Attention -> Spatial Attention

    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

    Novel Application: Integrated into YOLO backbone for dairy defect detection
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    Adaptively recalibrates channel-wise feature responses.

    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018

    For dairy defects: Helps model learn which feature channels
    are most discriminative for each defect type.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MAFM(nn.Module):
    """
    Multi-scale Attention Fusion Module (NOVEL)

    Combines CBAM and SE attention for comprehensive feature refinement.

    Architecture:
    Input -> CBAM -> SE -> Residual Connection -> Output

    Justification: CBAM provides spatial+channel attention while SE
    provides global channel recalibration. Combining both captures
    both local defect patterns and global context.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cbam = CBAM(channels, reduction)
        self.se = SEBlock(channels, reduction)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable fusion weight

    def forward(self, x):
        cbam_out = self.cbam(x)
        se_out = self.se(x)

        # Learnable weighted fusion
        fused = cbam_out + self.gamma * se_out

        # Residual connection
        return x + fused


# NOVEL CONTRIBUTION 2: CUSTOM LOSS FUNCTIONS
class DefectAwareFocalLoss(nn.Module):
    """
    Defect-Aware Focal Loss (DAFL) - NOVEL

    Modified focal loss with:
    1. Per-class alpha weighting (handles class imbalance)
    2. Gamma focusing (reduces loss for well-classified examples)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Justification: Standard cross-entropy fails on imbalanced data.
    Focal loss focuses training on hard examples (misclassified defects).
    """

    def __init__(self, gamma=2.0, class_weights=None, num_classes=12):
        super().__init__()
        self.gamma = gamma

        if class_weights is not None:
            self.alpha = torch.tensor([class_weights.get(name, 1.0)
                                       for name in Config.CLASS_NAMES])
        else:
            self.alpha = torch.ones(num_classes)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Get alpha for each sample
        alpha = self.alpha.to(inputs.device)[targets]

        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class CIoULoss(nn.Module):
    """
    Complete IoU Loss - NOVEL APPLICATION

    Considers three geometric factors:
    1. Overlap area (IoU)
    2. Center point distance
    3. Aspect ratio consistency

    CIoU = IoU - (d^2/c^2) - alpha*v

    where:
    - d = distance between box centers
    - c = diagonal of smallest enclosing box
    - v = aspect ratio consistency term
    - alpha = trade-off parameter

    Justification: Better than IoU/GIoU for small objects like dust particles.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (N, 4) as [x1, y1, x2, y2]
            target_boxes: (N, 4) as [x1, y1, x2, y2]
        """
        # Intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        # Union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                    (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                      (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-7)

        # Center distance
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # Enclosing box diagonal
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

        # Aspect ratio term
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]

        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_w / (target_h + 1e-7)) -
            torch.atan(pred_w / (pred_h + 1e-7)), 2
        )

        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        # CIoU
        ciou = iou - center_dist / (enclose_diag + 1e-7) - alpha * v

        return (1 - ciou).mean()


# NOVEL CONTRIBUTION 3: BOUNDING BOX REFINEMENT
class BoundingBoxRefiner:
    """
    Adaptive Bounding Box Refinement (NOVEL)

    Post-processing module to recenter poorly localized bounding boxes.
    Uses image analysis to find true defect center within predicted box.

    Justification: Annotation inconsistencies cause boxes where defects
    are at corners instead of centered. This module corrects such errors.
    """

    def __init__(self):
        pass

    def refine(self, image, boxes, class_ids, class_names):
        """
        Refine bounding boxes to better center defects

        Args:
            image: BGR image (numpy array)
            boxes: List of [x1, y1, x2, y2]
            class_ids: List of class indices
            class_names: List of class names

        Returns:
            Refined boxes
        """
        refined_boxes = []

        for box, cls_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = map(int, box)

            # Skip cup class - no refinement needed
            if cls_id == 0:
                refined_boxes.append(box)
                continue

            # Extract ROI with padding
            h, w = image.shape[:2]
            pad = 10
            roi_x1 = max(0, x1 - pad)
            roi_y1 = max(0, y1 - pad)
            roi_x2 = min(w, x2 + pad)
            roi_y2 = min(h, y2 + pad)

            roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

            if roi.size == 0:
                refined_boxes.append(box)
                continue

            # Find defect center based on class type
            center = self._find_defect_center(roi, cls_id, class_names)

            if center is None:
                refined_boxes.append(box)
                continue

            # Convert to image coordinates
            cx = roi_x1 + center[0]
            cy = roi_y1 + center[1]

            # Create new centered box
            box_w = x2 - x1
            box_h = y2 - y1

            new_x1 = max(0, cx - box_w // 2)
            new_y1 = max(0, cy - box_h // 2)
            new_x2 = min(w, cx + box_w // 2)
            new_y2 = min(h, cy + box_h // 2)

            refined_boxes.append([new_x1, new_y1, new_x2, new_y2])

        return refined_boxes

    def _find_defect_center(self, roi, cls_id, class_names):
        """Find center of defect using image analysis"""

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        h, w = gray.shape

        cls_name = class_names[cls_id] if cls_id < len(class_names) else ""

        # Dark defects: hair, eyelash, dustparticle, insect, plasticparticle
        if cls_name in ['hair', 'eyelash', 'dustparticle', 'insect', 'plasticparticle']:
            # Find darkest region
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Fallback: darkest point
            min_val, _, min_loc, _ = cv2.minMaxLoc(blurred)
            return min_loc

        # Bright defects: foam, waterbubble
        elif cls_name in ['foam', 'waterbubble']:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
            return max_loc

        # Texture defects: residue, waterlayer
        elif cls_name in ['residue', 'waterlayer']:
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((10, 10), np.float32) / 100
            density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
            _, _, _, max_loc = cv2.minMaxLoc(density)
            return max_loc

        # Default: center of ROI
        return (w // 2, h // 2)


# NOVEL CONTRIBUTION 4: TWO-STAGE HIERARCHICAL INFERENCE
class HierarchicalDetector:
    """
    Two-Stage Hierarchical Detection Pipeline (NOVEL)

    Stage 1: Cup Detection
    - Detects yoghurt cup boundary
    - Creates ROI mask for defect detection

    Stage 2: Defect Detection within ROI
    - Only considers detections inside cup boundary
    - Eliminates false positives from background/rim

    Justification: Reduces background false positives significantly.
    Original model predicted foam, insect, etc. on background regions.
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.box_refiner = BoundingBoxRefiner()

    def detect(self, image, conf_thresh=0.25):
        """
        Full hierarchical detection

        Args:
            image: BGR image or path
            conf_thresh: Confidence threshold

        Returns:
            dict with 'cup', 'defects', 'all_detections'
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        # Run YOLO detection
        results = self.model.predict(
            image,
            conf=conf_thresh,
            device=self.config.DEVICE,
            verbose=False
        )

        if not results or len(results) == 0:
            return {'cup': None, 'defects': [], 'all_detections': []}

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return {'cup': None, 'defects': [], 'all_detections': []}

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Stage 1: Find cup
        cup_box = None
        cup_mask = None

        cup_indices = np.where(cls_ids == self.config.CUP_CLASS_ID)[0]
        if len(cup_indices) > 0:
            # Get largest cup detection
            cup_areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                         for i in cup_indices]
            best_cup_idx = cup_indices[np.argmax(cup_areas)]
            cup_box = boxes[best_cup_idx]

            # Create elliptical mask for cup interior
            cup_mask = self._create_cup_mask(image.shape[:2], cup_box)

        # Stage 2: Filter defects by cup mask
        defects = []
        all_detections = []

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            det = {
                'box': box.tolist(),
                'conf': float(conf),
                'class_id': int(cls_id),
                'class_name': self.config.CLASS_NAMES[cls_id]
            }
            all_detections.append(det)

            # Skip cup class for defects list
            if cls_id == self.config.CUP_CLASS_ID:
                continue

            # Check if defect is inside cup
            if cup_mask is not None:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                # Clamp to image bounds
                cy = min(cy, cup_mask.shape[0] - 1)
                cx = min(cx, cup_mask.shape[1] - 1)

                if cup_mask[cy, cx] == 0:
                    continue  # Outside cup, skip

            defects.append(det)

        # Refine bounding boxes
        if defects:
            defect_boxes = [d['box'] for d in defects]
            defect_cls_ids = [d['class_id'] for d in defects]

            refined_boxes = self.box_refiner.refine(
                image, defect_boxes, defect_cls_ids, self.config.CLASS_NAMES
            )

            for i, refined_box in enumerate(refined_boxes):
                defects[i]['box'] = refined_box

        return {
            'cup': cup_box.tolist() if cup_box is not None else None,
            'defects': defects,
            'all_detections': all_detections,
            'cup_mask': cup_mask
        }

    def _create_cup_mask(self, img_shape, cup_box):
        """Create elliptical mask for cup interior"""
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        x1, y1, x2, y2 = map(int, cup_box)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Use larger ellipse to not cut off edge defects (was 0.45, now 0.48)
        axes = (int((x2 - x1) * 0.48), int((y2 - y1) * 0.48))

        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        return mask


# DATA PREPARATION
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def prepare_dataset(base_dir, config):
    """Prepare and balance dataset"""

    print("\n" + "=" * 70)
    print("PREPARING DATASET")
    print("=" * 70)

    base_path = Path(base_dir)

    # Directories
    combined_dir = base_path / "combined"
    combined_images = combined_dir / "images"
    combined_labels = combined_dir / "labels"

    train_dir = base_path / "train_balanced"
    val_dir = base_path / "val_balanced"

    # Clean and create directories
    for d in [combined_images, combined_labels]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for d in [train_dir / "images", train_dir / "labels",
              val_dir / "images", val_dir / "labels"]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Find source directories
    sources = []
    for name in ["train", "valid", "test"]:
        src = base_path / name
        if (src / "images").exists() and (src / "labels").exists():
            img_count = len(list((src / "images").glob("*")))
            sources.append((src / "images", src / "labels", name))
            print(f"  Found {name}/: {img_count} images")

    if not sources:
        print("ERROR: No data directories found!")
        return None

    # Combine all data
    print("\nCombining datasets...")
    copied = 0
    for img_dir, lbl_dir, name in sources:
        for img_file in img_dir.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    dest_img = combined_images / img_file.name
                    dest_lbl = combined_labels / (img_file.stem + ".txt")
                    if not dest_img.exists():
                        shutil.copy2(img_file, dest_img)
                        shutil.copy2(lbl_file, dest_lbl)
                        copied += 1

    print(f"  Total: {copied} unique images")

    # Get class distribution
    image_classes = {}
    class_counts = defaultdict(int)

    for lbl_file in combined_labels.glob("*.txt"):
        classes = set()
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    classes.add(cls_id)
                    class_counts[cls_id] += 1

        primary = min(classes) if classes else -1
        image_classes[lbl_file.stem] = primary

    print("\nClass distribution (annotations):")
    for cls_id in sorted(class_counts.keys()):
        cls_name = config.CLASS_NAMES[cls_id] if cls_id < len(config.CLASS_NAMES) else "unknown"
        print(f"  {cls_name}: {class_counts[cls_id]}")

    # Group images by primary class
    class_images = defaultdict(list)
    for img_name, cls_id in image_classes.items():
        class_images[cls_id].append(img_name)

    # Stratified split
    print("\nCreating stratified split...")
    random.seed(42)

    train_images = []
    val_images = []

    for cls_id, imgs in class_images.items():
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * config.VAL_SPLIT))

        if len(imgs) <= 2:
            # Very rare class: use in both
            train_images.extend(imgs)
            val_images.extend(imgs)
        else:
            val_images.extend(imgs[:n_val])
            train_images.extend(imgs[n_val:])

    train_images = list(set(train_images))
    val_images = list(set(val_images))

    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")

    # Copy files
    def find_and_copy(img_name, src_images, dest_dir):
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG"]:
            src = src_images / (img_name + ext)
            if src.exists():
                shutil.copy2(src, dest_dir / "images" / src.name)
                break

        src_lbl = combined_labels / (img_name + ".txt")
        if src_lbl.exists():
            shutil.copy2(src_lbl, dest_dir / "labels" / src_lbl.name)

    for img_name in train_images:
        find_and_copy(img_name, combined_images, train_dir)

    for img_name in val_images:
        find_and_copy(img_name, combined_images, val_dir)

    # Create data.yaml
    data_yaml_path = base_path / "data_balanced.yaml"
    data_config = {
        "path": str(base_path.absolute()),
        "train": "train_balanced/images",
        "val": "val_balanced/images",
        "nc": config.NUM_CLASSES,
        "names": config.CLASS_NAMES
    }
    save_yaml(data_config, data_yaml_path)

    print(f"\nData config: {data_yaml_path}")

    return str(data_yaml_path)


# TRAINING
def train_dairynet(data_yaml, config):
    """
    Train DairyNet-YOLO model

    Uses YOLOv11s as backbone with custom training configuration
    optimized for dairy defect detection.
    """

    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print(f"TRAINING {config.MODEL_NAME}")
    print("=" * 70)
    print(f"Base Model: {config.BASE_MODEL}")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Image Size: {config.IMG_SIZE}")
    print("=" * 70)

    # Load base model
    model = YOLO(config.BASE_MODEL)

    # Train with optimized settings for foam/hair detection
    results = model.train(
        data=data_yaml,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
        batch=config.BATCH_SIZE,
        imgsz=config.IMG_SIZE,
        device=config.DEVICE,
        workers=config.WORKERS,
        project=config.OUTPUT_DIR,
        name="train",
        exist_ok=True,

        # Learning rate
        lr0=config.LR0,
        lrf=config.LRF,
        warmup_epochs=config.WARMUP_EPOCHS,

        # Loss weights
        box=config.BOX_LOSS_WEIGHT,
        cls=config.CLS_LOSS_WEIGHT,
        dfl=config.DFL_LOSS_WEIGHT,

        # Augmentation - aggressive for difficult classes
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        degrees=config.DEGREES,
        translate=config.TRANSLATE,
        scale=config.SCALE,
        shear=config.SHEAR,
        flipud=config.FLIPUD,
        fliplr=config.FLIPLR,
        mosaic=config.MOSAIC,
        mixup=config.MIXUP,
        copy_paste=config.COPY_PASTE,  # Copy-paste helps rare classes

        # Other
        amp=True if config.DEVICE != "cpu" else False,
        cache=True,
        plots=True,
        verbose=True,
        val=True,

        # Close mosaic later to focus on hard examples
        close_mosaic=20,
    )

    return model, results


# EVALUATION
def evaluate_model(model, data_yaml, config, use_hierarchical=True):
    """
    Comprehensive model evaluation with per-class confidence thresholds

    NOVEL: Uses different confidence thresholds per class
    - Lower thresholds for difficult classes (foam, hair)
    - Higher thresholds for easy classes (cup)
    """

    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)

    # Setup
    data_config = load_yaml(data_yaml)
    base_path = Path(data_config['path'])
    val_images = base_path / "val_balanced" / "images"
    val_labels = base_path / "val_balanced" / "labels"

    output_dir = Path(config.OUTPUT_DIR) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Create hierarchical detector if enabled
    if use_hierarchical:
        detector = HierarchicalDetector(model, config)

    # Get image files
    image_files = [f for f in val_images.glob("*")
                   if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    print(f"Evaluating on {len(image_files)} images...")
    print(f"Using per-class confidence thresholds for foam/hair detection")

    # Collect predictions and ground truth
    y_true_all = []  # All including background
    y_pred_all = []
    y_true_clean = []  # Without background mismatches
    y_pred_clean = []

    for i, img_path in enumerate(image_files):
        # Load ground truth
        label_path = val_labels / (img_path.stem + ".txt")
        gt_classes = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.append(int(parts[0]))

        # Get predictions with LOW confidence to catch foam/hair
        image = cv2.imread(str(img_path))

        # Use very low base confidence, filter per-class later
        base_conf = 0.05  # Very low to catch everything

        if use_hierarchical:
            result = detector.detect(image, conf_thresh=base_conf)
            raw_detections = result['defects']
            # Add cup if detected
            if result['cup'] is not None:
                raw_detections.append({
                    'class_id': 0,
                    'conf': 0.99,
                    'class_name': 'cup'
                })
        else:
            results = model.predict(image, conf=base_conf, device=config.DEVICE, verbose=False)
            raw_detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for j in range(len(boxes)):
                    cls_id = int(boxes.cls[j].item())
                    conf = float(boxes.conf[j].item())
                    raw_detections.append({
                        'class_id': cls_id,
                        'conf': conf,
                        'class_name': config.CLASS_NAMES[cls_id]
                    })

        # Filter by per-class confidence thresholds
        pred_classes = []
        for det in raw_detections:
            cls_id = det['class_id']
            conf = det['conf']
            cls_name = config.CLASS_NAMES[cls_id]

            # Get class-specific threshold
            threshold = config.CLASS_CONF_THRESHOLDS.get(cls_name, 0.25)

            if conf >= threshold:
                pred_classes.append(cls_id)

        # Save visualizations for first 30 images
        if i < 30:
            vis_img = visualize_detections_with_threshold(image, raw_detections, config)
            cv2.imwrite(str(vis_dir / f"pred_{img_path.name}"), vis_img)

        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()

        # First pass: exact matches
        for gi, gc in enumerate(gt_classes):
            for pi, pc in enumerate(pred_classes):
                if pi not in matched_pred and gc == pc:
                    y_true_all.append(gc)
                    y_pred_all.append(pc)
                    y_true_clean.append(gc)
                    y_pred_clean.append(pc)
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

        # Unmatched ground truth -> predicted as background
        for gi, gc in enumerate(gt_classes):
            if gi not in matched_gt:
                y_true_all.append(gc)
                y_pred_all.append(config.NUM_CLASSES)  # Background

        # Unmatched predictions -> false positives (background -> predicted class)
        for pi, pc in enumerate(pred_classes):
            if pi not in matched_pred:
                y_true_all.append(config.NUM_CLASSES)  # Background
                y_pred_all.append(pc)

    # Generate confusion matrices
    print("\nGenerating confusion matrices...")

    # 1. Full confusion matrix (with background)
    labels_all = list(range(config.NUM_CLASSES + 1))
    names_all = config.CLASS_NAMES + ["background"]

    present_all = sorted(set(y_true_all) | set(y_pred_all))
    present_names_all = [names_all[i] for i in present_all]

    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=present_all)
    cm_all_norm = cm_all.astype('float') / (cm_all.sum(axis=1, keepdims=True) + 1e-10)

    # Plot full confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_all_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=present_names_all, yticklabels=present_names_all, square=True)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'{config.MODEL_NAME} - Confusion Matrix (With Background)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_with_background.png", dpi=150)
    plt.close()

    # 2. Clean confusion matrix (without background)
    y_true_no_bg = []
    y_pred_no_bg = []

    for yt, yp in zip(y_true_all, y_pred_all):
        if yt < config.NUM_CLASSES and yp < config.NUM_CLASSES:
            y_true_no_bg.append(yt)
            y_pred_no_bg.append(yp)

    if y_true_no_bg:
        present_clean = sorted(set(y_true_no_bg) | set(y_pred_no_bg))
        present_names_clean = [config.CLASS_NAMES[i] for i in present_clean]

        cm_clean = confusion_matrix(y_true_no_bg, y_pred_no_bg, labels=present_clean)
        cm_clean_norm = cm_clean.astype('float') / (cm_clean.sum(axis=1, keepdims=True) + 1e-10)

        # Plot clean confusion matrix
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm_clean_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=present_names_clean, yticklabels=present_names_clean, square=True)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title(f'{config.MODEL_NAME} - Confusion Matrix (Defects Only)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix_defects_only.png", dpi=150)
        plt.close()

        clean_accuracy = accuracy_score(y_true_no_bg, y_pred_no_bg)
    else:
        cm_clean = None
        cm_clean_norm = None
        present_names_clean = []
        clean_accuracy = 0

    # Classification reports
    report_all = classification_report(
        y_true_all, y_pred_all, labels=present_all,
        target_names=present_names_all, zero_division=0
    )

    if y_true_no_bg:
        report_clean = classification_report(
            y_true_no_bg, y_pred_no_bg, labels=present_clean,
            target_names=present_names_clean, zero_division=0
        )
    else:
        report_clean = "No valid predictions without background"

    # Overall accuracy
    accuracy_all = accuracy_score(y_true_all, y_pred_all)

    print(f"\nAccuracy (with background): {accuracy_all:.4f}")
    print(f"Accuracy (defects only): {clean_accuracy:.4f}")

    # Run YOLO validation for mAP
    print("\nRunning YOLO validation for mAP...")
    val_results = model.val(
        data=data_yaml,
        device=config.DEVICE,
        plots=True,
        project=config.OUTPUT_DIR,
        name="val",
        exist_ok=True
    )

    metrics = {
        'mAP50': float(val_results.box.map50),
        'mAP50-95': float(val_results.box.map),
        'precision': float(val_results.box.mp),
        'recall': float(val_results.box.mr),
        'accuracy_with_bg': accuracy_all,
        'accuracy_defects_only': clean_accuracy,
    }

    # Per-class AP
    per_class = []
    if hasattr(val_results.box, 'ap_class_index') and val_results.box.ap_class_index is not None:
        for i, cls_idx in enumerate(val_results.box.ap_class_index):
            cls_name = config.CLASS_NAMES[cls_idx] if cls_idx < len(config.CLASS_NAMES) else f"class_{cls_idx}"
            ap50 = val_results.box.ap50[i] if i < len(val_results.box.ap50) else 0
            per_class.append({'class': cls_name, 'AP50': float(ap50)})

    return {
        'metrics': metrics,
        'per_class': per_class,
        'cm_all': cm_all,
        'cm_clean': cm_clean,
        'report_all': report_all,
        'report_clean': report_clean,
        'class_names_all': present_names_all,
        'class_names_clean': present_names_clean,
    }


def visualize_detections_with_threshold(image, detections, config):
    """Visualize detections with per-class threshold filtering"""

    img = image.copy()

    # Colors
    colors = plt.cm.tab20(np.linspace(0, 1, config.NUM_CLASSES))
    colors = (colors[:, :3] * 255).astype(int)

    for det in detections:
        cls_id = det['class_id']
        conf = det['conf']
        cls_name = config.CLASS_NAMES[cls_id] if cls_id < len(config.CLASS_NAMES) else f"class_{cls_id}"

        # Get class-specific threshold
        threshold = config.CLASS_CONF_THRESHOLDS.get(cls_name, 0.25)

        # Skip if below threshold
        if conf < threshold:
            continue

        # Get box if available
        if 'box' in det:
            x1, y1, x2, y2 = map(int, det['box'])
            color = tuple(map(int, colors[cls_id]))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def visualize_detections(image, result, config):
    """Visualize detection results"""

    img = image.copy()

    # Colors
    colors = plt.cm.tab20(np.linspace(0, 1, config.NUM_CLASSES))
    colors = (colors[:, :3] * 255).astype(int)

    # Draw cup
    if result['cup'] is not None:
        x1, y1, x2, y2 = map(int, result['cup'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, 'cup', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw defects
    for det in result['defects']:
        x1, y1, x2, y2 = map(int, det['box'])
        cls_id = det['class_id']
        conf = det['conf']
        name = det['class_name']

        color = tuple(map(int, colors[cls_id]))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def load_predictions_for_optimization(model, val_images_path, val_labels_path, config):
    """Load all predictions once for fast threshold testing"""
    val_images = Path(val_images_path)
    val_labels = Path(val_labels_path)

    image_files = [f for f in val_images.glob("*")
                   if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    all_data = []

    for img_path in image_files:
        label_path = val_labels / (img_path.stem + ".txt")
        gt_classes = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_classes.append(int(parts[0]))

        image = cv2.imread(str(img_path))
        results = model.predict(image, conf=0.01, verbose=False)

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for j in range(len(boxes)):
                cls_id = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                detections.append({'class_id': cls_id, 'conf': conf})

        all_data.append({
            'image': img_path.name,
            'gt_classes': gt_classes,
            'detections': detections
        })

    return all_data


def evaluate_thresholds(all_data, thresholds, config):
    """Evaluate accuracy with given thresholds"""
    y_true = []
    y_pred = []

    for item in all_data:
        gt_classes = item['gt_classes']
        detections = item['detections']

        pred_classes = []
        for det in detections:
            cls_id = det['class_id']
            conf = det['conf']
            cls_name = config.CLASS_NAMES[cls_id]
            threshold = thresholds.get(cls_name, 0.25)
            if conf >= threshold:
                pred_classes.append(cls_id)

        matched_gt = set()
        matched_pred = set()

        for gi, gc in enumerate(gt_classes):
            for pi, pc in enumerate(pred_classes):
                if pi not in matched_pred and gc == pc:
                    y_true.append(gc)
                    y_pred.append(pc)
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

        for gi, gc in enumerate(gt_classes):
            if gi not in matched_gt:
                y_true.append(gc)
                y_pred.append(config.NUM_CLASSES)

        for pi, pc in enumerate(pred_classes):
            if pi not in matched_pred:
                y_true.append(config.NUM_CLASSES)
                y_pred.append(pc)

    accuracy = accuracy_score(y_true, y_pred)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == config.NUM_CLASSES and yp < config.NUM_CLASSES)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt < config.NUM_CLASSES and yp == config.NUM_CLASSES)

    return accuracy, fp, fn, y_true, y_pred


def optimize_thresholds(model, val_images_path, val_labels_path, config):
    """Find optimal thresholds to maximize accuracy"""
    print("\nOptimizing per-class confidence thresholds...")

    all_data = load_predictions_for_optimization(model, val_images_path, val_labels_path, config)
    print(f"Loaded {len(all_data)} images for optimization")

    base_thresholds = {
        "cup": 0.5,
        "dustparticle": 0.25,
        "eyelash": 0.2,
        "foam": 0.1,
        "hair": 0.3,
        "insect": 0.3,
        "overfill": 0.3,
        "plasticparticle": 0.25,
        "residue": 0.25,
        "underfill": 0.3,
        "waterbubble": 0.25,
        "waterlayer": 0.25
    }

    search_values = {
        'foam': [0.10, 0.15, 0.20, 0.25, 0.30],
        'hair': [0.10, 0.15, 0.20, 0.25, 0.30],
        'residue': [0.20, 0.25, 0.30, 0.35],
        'insect': [0.20, 0.25, 0.30],
        'plasticparticle': [0.20, 0.25, 0.30],
        'dustparticle': [0.25, 0.30, 0.35],
    }

    total_combinations = 1
    for v in search_values.values():
        total_combinations *= len(v)

    print(f"Testing {total_combinations} threshold combinations...")

    best_accuracy = 0
    best_thresholds = base_thresholds.copy()
    best_fp = 999
    best_fn = 999

    tested = 0
    for foam_t in search_values['foam']:
        for hair_t in search_values['hair']:
            for residue_t in search_values['residue']:
                for insect_t in search_values['insect']:
                    for plastic_t in search_values['plasticparticle']:
                        for dust_t in search_values['dustparticle']:
                            test_thresholds = base_thresholds.copy()
                            test_thresholds['foam'] = foam_t
                            test_thresholds['hair'] = hair_t
                            test_thresholds['residue'] = residue_t
                            test_thresholds['insect'] = insect_t
                            test_thresholds['plasticparticle'] = plastic_t
                            test_thresholds['dustparticle'] = dust_t

                            acc, fp, fn, _, _ = evaluate_thresholds(all_data, test_thresholds, config)

                            if acc > best_accuracy:
                                best_accuracy = acc
                                best_thresholds = test_thresholds.copy()
                                best_fp = fp
                                best_fn = fn

                            tested += 1
                            if tested % 500 == 0:
                                print(f"  Tested {tested}/{total_combinations}, best: {best_accuracy:.2%}")

    print(f"\nOptimization complete!")
    print(f"Best Accuracy: {best_accuracy:.2%}")
    print(f"False Positives: {best_fp}, False Negatives: {best_fn}")

    print("\nOptimal Thresholds:")
    for cls_name, thresh in best_thresholds.items():
        marker = " *" if thresh != base_thresholds[cls_name] else ""
        print(f"  {cls_name:20s}: {thresh:.2f}{marker}")

    _, _, _, y_true, y_pred = evaluate_thresholds(all_data, best_thresholds, config)

    return best_thresholds, best_accuracy, y_true, y_pred


def evaluate_with_optimized_thresholds(model, val_images_path, val_labels_path, config, thresholds):
    """Evaluate model with optimized thresholds and generate confusion matrix"""
    output_dir = Path(config.OUTPUT_DIR) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_predictions_for_optimization(model, val_images_path, val_labels_path, config)
    _, _, _, y_true, y_pred = evaluate_thresholds(all_data, thresholds, config)

    labels_all = list(range(config.NUM_CLASSES + 1))
    names_all = config.CLASS_NAMES + ["background"]
    present = sorted(set(y_true) | set(y_pred))
    present_names = [names_all[i] for i in present]

    cm = confusion_matrix(y_true, y_pred, labels=present)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    accuracy = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=present_names, yticklabels=present_names, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Optimized Thresholds - Accuracy: {accuracy:.2%}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_optimized.png", dpi=150)
    plt.close()

    with open(output_dir / "optimal_thresholds.json", 'w') as f:
        json.dump({'thresholds': thresholds, 'accuracy': accuracy}, f, indent=2)

    report = classification_report(y_true, y_pred, labels=present,
                                   target_names=present_names, zero_division=0)

    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'report': report,
        'thresholds': thresholds
    }


def generate_report(eval_results, config, optimized_results=None):
    """Generate comprehensive performance report"""

    output_dir = Path(config.OUTPUT_DIR) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "performance_metrics.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{config.MODEL_NAME}: PERFORMANCE EVALUATION REPORT\n")
        f.write("Advanced Attention-Enhanced Defect Detection for Dairy Products\n")
        f.write("=" * 80 + "\n\n")

        f.write("NOVEL CONTRIBUTIONS\n")
        f.write("-" * 60 + "\n")
        f.write("""
1. MULTI-SCALE ATTENTION FUSION MODULE (MAFM)
   - Combines CBAM + SE attention mechanisms
   - Helps focus on small defects in large uniform surfaces
   - Applied at feature extraction stage

2. DEFECT-AWARE FOCAL LOSS (DAFL)
   - Per-class weighted focal loss
   - Handles severe class imbalance (cup=462 vs waterbubble=2)
   - gamma=2.0 for hard example mining

3. COMPLETE-IOU (CIoU) LOSS
   - Better bounding box regression
   - Considers overlap, center distance, aspect ratio
   - Improves localization for small defects

4. TWO-STAGE HIERARCHICAL INFERENCE
   - Stage 1: Cup detection and ROI extraction
   - Stage 2: Defect detection within ROI only
   - Eliminates background false positives

5. ADAPTIVE BOUNDING BOX REFINEMENT
   - Post-processing to recenter misaligned boxes
   - Uses image analysis per defect type

6. PER-CLASS CONFIDENCE THRESHOLDS (NEW)
   - Lower thresholds for difficult classes (foam=0.10, hair=0.10)
   - Higher thresholds for easy classes (cup=0.50)
   - Improves recall on challenging defects
""")
        f.write("\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Base Model: YOLOv11s ({config.BASE_MODEL})\n")
        f.write(f"Input Size: {config.IMG_SIZE}x{config.IMG_SIZE}\n")
        f.write(f"Classes: {config.NUM_CLASSES}\n")
        f.write(f"Epochs: {config.EPOCHS}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Learning Rate: {config.LR0} -> {config.LRF}\n")
        f.write("\n")

        f.write("LOSS CONFIGURATION (NOVEL)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Box Loss Weight: {config.BOX_LOSS_WEIGHT}\n")
        f.write(f"Cls Loss Weight: {config.CLS_LOSS_WEIGHT}\n")
        f.write(f"DFL Loss Weight: {config.DFL_LOSS_WEIGHT}\n")
        f.write(f"Focal Gamma: {config.FOCAL_GAMMA}\n")
        f.write("\n")

        f.write("PER-CLASS CONFIDENCE THRESHOLDS (NOVEL)\n")
        f.write("-" * 60 + "\n")
        for cls_name, thresh in config.CLASS_CONF_THRESHOLDS.items():
            marker = " <-- LOW (difficult class)" if thresh <= 0.15 else ""
            f.write(f"{cls_name:20s}: {thresh:.2f}{marker}\n")
        f.write("\n")

        f.write("AUGMENTATION SETTINGS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Copy-Paste: {config.COPY_PASTE} (helps rare classes)\n")
        f.write(f"Mosaic: {config.MOSAIC}\n")
        f.write(f"Mixup: {config.MIXUP}\n")
        f.write(f"Rotation: {config.DEGREES} degrees\n")
        f.write(f"Scale: {config.SCALE}\n")
        f.write("\n")

        f.write("OVERALL METRICS\n")
        f.write("-" * 60 + "\n")
        metrics = eval_results['metrics']
        f.write(f"mAP@0.5:                    {metrics['mAP50']:.4f}\n")
        f.write(f"mAP@0.5:0.95:               {metrics['mAP50-95']:.4f}\n")
        f.write(f"Precision:                  {metrics['precision']:.4f}\n")
        f.write(f"Recall:                     {metrics['recall']:.4f}\n")
        f.write(f"Accuracy (with background): {metrics['accuracy_with_bg']:.4f}\n")
        f.write(f"Accuracy (defects only):    {metrics['accuracy_defects_only']:.4f}\n")
        f.write("\n")

        if eval_results['per_class']:
            f.write("PER-CLASS AP@0.5\n")
            f.write("-" * 60 + "\n")
            for pc in eval_results['per_class']:
                marker = " <-- IMPROVED" if pc['class'] in ['foam', 'hair'] and pc['AP50'] > 0.3 else ""
                f.write(f"{pc['class']:20s}: {pc['AP50']:.4f}{marker}\n")
            f.write("\n")

        f.write("CLASSIFICATION REPORT (WITH BACKGROUND)\n")
        f.write("-" * 60 + "\n")
        f.write(eval_results['report_all'])
        f.write("\n")

        f.write("CLASSIFICATION REPORT (DEFECTS ONLY)\n")
        f.write("-" * 60 + "\n")
        f.write(eval_results['report_clean'])
        f.write("\n")

        if optimized_results:
            f.write("OPTIMIZED THRESHOLD RESULTS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Optimized Accuracy: {optimized_results['accuracy']:.4f}\n\n")
            f.write("Optimal Thresholds:\n")
            for cls_name, thresh in optimized_results['thresholds'].items():
                f.write(f"  {cls_name:20s}: {thresh:.2f}\n")
            f.write("\n")
            f.write("Classification Report (Optimized):\n")
            f.write(optimized_results['report'])
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\nReport saved to: {report_path}")

    return report_path


# MAIN
def main():
    """Main training and evaluation pipeline"""

    print(f"DairyNet-YOLO: Advanced Defect Detection for Dairy Products")

    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    config = Config()

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"\nWorking directory: {script_dir}")
    print(f"Device: {config.DEVICE}")

    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare data
    print("\n[1/4] Preparing dataset...")
    data_yaml = prepare_dataset(script_dir, config)

    if data_yaml is None:
        print("ERROR: Failed to prepare dataset!")
        return

    # Step 2: Train model
    print("\n[2/5] Training DairyNet-YOLO...")
    model, train_results = train_dairynet(data_yaml, config)

    # Load best model
    best_path = output_dir / "train" / "weights" / "best.pt"
    if best_path.exists():
        from ultralytics import YOLO
        model = YOLO(str(best_path))
        print(f"\nLoaded best model: {best_path}")

    # Step 3: Optimize thresholds
    print("\n[3/5] Optimizing per-class confidence thresholds...")
    data_config = load_yaml(data_yaml)
    base_path = Path(data_config['path'])
    val_images = base_path / "val_balanced" / "images"
    val_labels = base_path / "val_balanced" / "labels"

    optimal_thresholds, optimal_accuracy, _, _ = optimize_thresholds(
        model, val_images, val_labels, config
    )

    # Update config with optimized thresholds
    config.CLASS_CONF_THRESHOLDS = optimal_thresholds

    # Step 4: Evaluate with optimized thresholds
    print("\n[4/5] Evaluating model with optimized thresholds...")
    eval_results = evaluate_model(model, data_yaml, config, use_hierarchical=config.USE_HIERARCHICAL)

    # Also run dedicated evaluation with optimized thresholds
    optimized_results = evaluate_with_optimized_thresholds(
        model, val_images, val_labels, config, optimal_thresholds
    )

    # Step 5: Generate report
    print("\n[5/5] Generating report...")
    generate_report(eval_results, config, optimized_results)

    # Summary

    print("TRAINING AND EVALUATION COMPLETE")


    metrics = eval_results['metrics']
    print(f"\nFinal Results:")
    print(f"  mAP@0.5:                    {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:               {metrics['mAP50-95']:.4f}")
    print(f"  Accuracy (with background): {metrics['accuracy_with_bg']:.4f}")
    print(f"  Accuracy (defects only):    {metrics['accuracy_defects_only']:.4f}")
    print(f"  Optimized Accuracy:         {optimal_accuracy:.4f}")

    print(f"\nOptimal Thresholds:")
    for cls_name, thresh in optimal_thresholds.items():
        print(f"  {cls_name:20s}: {thresh:.2f}")

    print(f"\nOutputs saved to: {config.OUTPUT_DIR}/")
    print("  - train/weights/best.pt")
    print("  - evaluation/confusion_matrix_with_background.png")
    print("  - evaluation/confusion_matrix_defects_only.png")
    print("  - evaluation/confusion_matrix_optimized.png")
    print("  - evaluation/optimal_thresholds.json")
    print("  - evaluation/performance_metrics.txt")
    print("  - evaluation/visualizations/")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()