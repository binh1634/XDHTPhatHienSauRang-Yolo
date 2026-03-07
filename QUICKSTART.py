"""
Quick start guide for dental cavity detection system
"""

# ============================================================
# 1. ENVIRONMENT SETUP
# ============================================================

# Create virtual environment
# python -m venv venv
# venv\Scripts\activate

# Install dependencies
# pip install -r requirements.txt

# ============================================================
# 2. DATA PREPARATION
# ============================================================

from src.data_prep import create_data_splits

# Organize your X-ray images:
# data/raw/
# ├── images/          (X-ray files)
# ├── annotations/     (YOLO labels .txt)
# └── masks/           (Segmentation masks .png)

raw_data_dir = "data/raw"
processed_data_dir = "data/processed"

create_data_splits(raw_data_dir, processed_data_dir)

# ============================================================
# 3. TRAINING
# ============================================================

from src.training import train_unet_from_config
from src.yolo_model import YOLODetector

# Train UNet for segmentation
train_unet_from_config(config_path="config.yaml")

# Train YOLO for detection (requires data.yaml)
# yolo detect train data=data.yaml model=yolov8m.pt epochs=100 imgsz=640

# ============================================================
# 4. INFERENCE
# ============================================================

from src.inference import DentalCavityInference

# Initialize inference system
inference = DentalCavityInference(
    yolo_weights="models/yolo_weights.pt",
    unet_weights="models/best_unet.pt",
    device="cuda"
)

# Process single image
results = inference.process_image("data/test/image.jpg", save_output=True)
print(f"Found {results['num_cavities']} cavities")

# ============================================================
# 5. EVALUATION
# ============================================================

from src.evaluation import Evaluator

evaluator = Evaluator(task="both")

# Evaluate segmentation
seg_metrics = evaluator.evaluate_segmentation(
    predictions=predictions_list,
    targets=targets_list
)

# Evaluate detection
det_metrics = evaluator.evaluate_detection(
    predictions=pred_boxes,
    ground_truths=gt_boxes
)

print("="*50)
print("RESULTS")
print("="*50)
print(f"Mean Dice: {seg_metrics['mean_dice']:.4f}")
print(f"Mean IoU: {seg_metrics['mean_iou']:.4f}")
print(f"mAP: {det_metrics['mAP']:.4f}")
print("="*50)

# ============================================================
# CONFIGURATION
# ============================================================

# Edit config.yaml to adjust:
# - Learning rate, batch size, epochs
# - Model architecture (UNet depth, features)
# - YOLO model size (n, s, m, l, x)
# - Data augmentation settings
# - Paths to data and models
