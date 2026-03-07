"""
Utility functions for the dental cavity detection system
"""

import os
import json
import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise


def create_directories(config: Dict) -> None:
    """Create necessary directories from config"""
    paths = config.get('paths', {})
    for path_name, path_value in paths.items():
        Path(path_value).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path_value}")


def save_json(data: Dict, filepath: str) -> None:
    """Save dictionary to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [0, 1] to [0, 255]"""
    return (image * 255).astype(np.uint8)


def resize_image(image: np.ndarray, size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """Resize image to specified size"""
    if isinstance(size, int):
        size = (size, size)
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance X-ray"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_8bit = denormalize_image(image) if image.max() <= 1.0 else image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(image_8bit)
    return normalize_image(enhanced)


def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                           sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter to denoise X-ray while preserving edges"""
    image_8bit = denormalize_image(image) if image.max() <= 1.0 else image.astype(np.uint8)
    if len(image_8bit.shape) == 2:
        filtered = cv2.bilateralFilter(image_8bit, d, sigma_color, sigma_space)
    else:
        filtered = cv2.bilateralFilter(image_8bit, d, sigma_color, sigma_space)
    return normalize_image(filtered)


def yolo_to_coco_bbox(yolo_bbox: List[float], image_width: int, 
                      image_height: int) -> List[int]:
    """
    Convert YOLO format bbox to COCO format
    YOLO: [center_x, center_y, width, height] (normalized 0-1)
    COCO: [x_min, y_min, width, height] (pixels)
    """
    cx, cy, w, h = yolo_bbox
    x_min = int((cx - w/2) * image_width)
    y_min = int((cy - h/2) * image_height)
    width = int(w * image_width)
    height = int(h * image_height)
    return [x_min, y_min, width, height]


def coco_to_yolo_bbox(coco_bbox: List[int], image_width: int, 
                      image_height: int) -> List[float]:
    """Convert COCO format bbox to YOLO format"""
    x_min, y_min, width, height = coco_bbox
    cx = (x_min + width/2) / image_width
    cy = (y_min + height/2) / image_height
    w = width / image_width
    h = height / image_height
    return [cx, cy, w, h]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU (Intersection over Union) between two boxes in YOLO format"""
    def get_corners(bbox):
        cx, cy, w, h = bbox
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    
    x1_min, y1_min, x1_max, y1_max = get_corners(box1)
    x2_min, y2_min, x2_max, y2_max = get_corners(box2)
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0
