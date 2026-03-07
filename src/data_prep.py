"""
Data preparation script for dental X-ray images
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_data_splits(raw_dir: str, processed_dir: str,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15):
    """
    Create train/val/test splits
    
    Expected raw_dir structure:
    raw_dir/
    ├── images/
    └── annotations/
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (processed_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (processed_path / split / 'yolo_labels').mkdir(parents=True, exist_ok=True)
        (processed_path / split / 'segmentation_masks').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_dir = raw_path / 'images'
    images = sorted([f for f in image_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    
    # Split
    n_images = len(images)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    
    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]
    
    logger.info(f"Total images: {n_images}")
    logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Copy files
    for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        for img_path in tqdm(split_images, desc=f"Processing {split_name}"):
            # Copy image
            dst_img = processed_path / split_name / 'images' / img_path.name
            if not dst_img.exists():
                import shutil
                shutil.copy(img_path, dst_img)
            
            # Copy label if exists
            label_path = raw_path / 'annotations' / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = processed_path / split_name / 'yolo_labels' / label_path.name
                if not dst_label.exists():
                    import shutil
                    shutil.copy(label_path, dst_label)
            
            # Copy mask if exists
            mask_path = raw_path / 'masks' / f"{img_path.stem}.png"
            if mask_path.exists():
                dst_mask = processed_path / split_name / 'segmentation_masks' / mask_path.name
                if not dst_mask.exists():
                    import shutil
                    shutil.copy(mask_path, dst_mask)
    
    logger.info("Data split completed!")


def enhance_xray_image(image_path: str, output_path: str, apply_clahe: bool = True):
    """Enhance X-ray image quality"""
    
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning(f"Failed to read: {image_path}")
        return
    
    # Apply CLAHE
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    
    # Save
    cv2.imwrite(output_path, image)


if __name__ == "__main__":
    # Example usage
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    
    # Create splits
    if Path(raw_data_dir).exists():
        create_data_splits(raw_data_dir, processed_data_dir)
    else:
        logger.warning(f"Raw data directory not found: {raw_data_dir}")
