"""
Data loading and preprocessing for dental cavity detection
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torch
import logging

logger = logging.getLogger(__name__)


class DentalXrayDataset(Dataset):
    """
    Dataset class for dental X-ray images
    
    Expected directory structure:
    data/processed/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── yolo_labels/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── segmentation_masks/
        ├── image1.png
        ├── image2.png
        └── ...
    """
    
    def __init__(self, 
                 data_dir: str,
                 image_size: int = 640,
                 task: str = "both",  # "detection", "segmentation", or "both"
                 augment: bool = False):
        """
        Args:
            data_dir: Path to processed data directory
            image_size: Target image size
            task: Type of task (detection, segmentation, or both)
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.label_dir = self.data_dir / "yolo_labels"
        self.mask_dir = self.data_dir / "segmentation_masks"
        self.image_size = image_size
        self.task = task
        
        # Get list of images
        self.image_files = sorted([f for f in self.image_dir.glob("*") 
                                   if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        
        if not self.image_files:
            logger.warning(f"No images found in {self.image_dir}")
        
        self.augmentation = self._get_augmentation() if augment else None
        
    def _get_augmentation(self):
        """Get data augmentation pipeline"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(p=0.2),
            A.GaussBlur(p=0.1),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image_name = image_path.stem
        
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        data = {
            'image': image,
            'image_name': image_name,
        }
        
        # Load detection labels if task includes detection
        if self.task in ["detection", "both"]:
            labels_path = self.label_dir / f"{image_name}.txt"
            bboxes, class_ids = self._load_yolo_labels(labels_path)
            data['bboxes'] = bboxes
            data['class_ids'] = class_ids
        
        # Load segmentation masks if task includes segmentation
        if self.task in ["segmentation", "both"]:
            mask_path = self.mask_dir / f"{image_name}.png"
            mask = self._load_mask(mask_path)
            data['mask'] = mask
        
        # Apply augmentation
        if self.augmentation is not None and 'bboxes' in data:
            augmented = self.augmentation(
                image=image,
                bboxes=data['bboxes'],
                class_labels=data['class_ids']
            )
            data['image'] = augmented['image']
            data['bboxes'] = augmented['bboxes']
            data['class_ids'] = augmented['class_labels']
        
        # Resize image
        data['image'] = cv2.resize(data['image'], (self.image_size, self.image_size))
        
        if 'mask' in data:
            data['mask'] = cv2.resize(data['mask'], (self.image_size, self.image_size))
        
        # Convert to tensors
        data['image'] = torch.from_numpy(data['image']).unsqueeze(0)  # Add channel dimension
        
        if 'mask' in data:
            data['mask'] = torch.from_numpy(data['mask']).unsqueeze(0)
        
        if 'bboxes' in data:
            data['bboxes'] = torch.tensor(data['bboxes'], dtype=torch.float32)
            data['class_ids'] = torch.tensor(data['class_ids'], dtype=torch.long)
        
        return data
    
    def _load_yolo_labels(self, labels_path: Path) -> Tuple[List, List]:
        """Load YOLO format labels"""
        bboxes = []
        class_ids = []
        
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        class_ids.append(class_id)
                        bboxes.append(bbox)
        
        return bboxes, class_ids
    
    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load segmentation mask"""
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask.astype(np.float32) / 255.0
        
        # Return empty mask if not found
        return np.zeros((self.image_size, self.image_size), dtype=np.float32)


def create_data_loaders(data_dir: str,
                       batch_size: int = 16,
                       image_size: int = 640,
                       task: str = "both",
                       num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    train_dataset = DentalXrayDataset(
        data_dir=f"{data_dir}/train",
        image_size=image_size,
        task=task,
        augment=True
    )
    
    val_dataset = DentalXrayDataset(
        data_dir=f"{data_dir}/val",
        image_size=image_size,
        task=task,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
