# YOLOv8 Lung Cancer Detection

## I. Introduction

This project focuses on building and training a **YOLOv8-based object detection model** for the task of **lung cancer (lung nodule) detection** on medical imaging data (CT scans / X-ray images).

The main objective is to apply **deep learning and computer vision techniques** to automatically **localize suspicious lung nodules**, supporting **early screening and preliminary diagnosis** in medical imaging workflows.

> **Disclaimer**:  
> This project is intended **for educational and research purposes only** and **must not be used as a substitute for professional medical diagnosis**.

---

## II. Objectives

- Train a YOLOv8 model for lung tumor / lung nodule detection  
- Evaluate model performance using standard object detection metrics  
- Analyze prediction results via **Confusion Matrix** and visual inspection  
- Demonstrate the feasibility of applying object detection models in medical imaging

---

## III. Model & Technologies

### Model
- **YOLOv8 (Ultralytics)** – one-stage object detection model  
  - Fast inference
  - High accuracy
  - Suitable for real-world deployment

### Libraries & Tools
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Matplotlib
- Google Colab Notebook

---

## IV. Dataset

- **Source**: Kaggle  
  *Lung Nodules Detection Dataset (with annotations)*  
- **Imaging modality**: Lung CT scans  
- **Annotation format**: YOLO bounding box format  
- **Dataset split**:
  - Training set  
  - Validation set  

### Classes
- `class_0`: No lung tumor detected  
- `class_1`: Lung tumor detected  
- `background`: Background / non-target regions  

---

## V. Training Configuration

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    workers=2
)
