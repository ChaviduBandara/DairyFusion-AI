# DairyFusion AI 🥛

## Multimodal Yoghurt Cup Defect Detection Using Hierarchical 
## Dual-Fusion of Visual Deep Learning and IoT Sensor Data

---

## Overview
DairyFusion AI is an end-to-end multimodal AI system that 
automatically detects yoghurt cup surface defects in real time 
by combining visual deep learning with live IoT sensor data. 
The system detects 12 defect classes across three categories, 
provides root cause explanations, and recommends corrective 
actions for factory operators.

---

## Three AI Models
- DairyNet YOLO V7 — YOLOv11s with 5 novel contributions — mAP@0.5 = 0.793
- DairyNet Multimodal V4 — Cross-attention hybrid meta-learner — Accuracy = 98.55%
- Sensor Classifier — Random Forest — Test Accuracy = 83.33%

---

## Five Novel YOLO Contributions
1. MAFM — Multi-Scale Attention Fusion Module
2. DAFL — Defect-Aware Focal Loss
3. CIoU Loss — Complete Intersection over Union
4. Two-Stage Hierarchical Inference
5. Per-Class Confidence Threshold Optimisation

---

## Defect Classes (12)
Physical Contaminants: hair, eyelash, insect, dustparticle, plasticparticle
Liquid Anomalies: foam, waterlayer, waterbubble, residue
Fill Level: overfill, underfill
No Defect: cup

---

## Tech Stack
- Backend: Python, FastAPI, PyTorch, Scikit-learn
- Frontend: React, JavaScript, Fetch API
- Hardware: ESP32-S3, HTU21D, DS18B20
- External API: Open-Meteo

---

## Dataset
First publicly available yoghurt cup defect dataset
462 images | 12 classes | Real factory data
Published on Kaggle: https://www.kaggle.com/datasets/chavidubandara/dairyfusion-ai-yoghurt-cup-defect-dataset

---

## Author
Chavidu Bandara | W1953563
IIT affiliated with University of Westminster
Supervised by Mr. Nipuna Senanayake
