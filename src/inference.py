"""
Inference module for dental cavity detection
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from src.yolo_model import YOLODetector
from src.unet_model import UNet
from src.utils import normalize_image, denormalize_image, apply_clahe

logger = logging.getLogger(__name__)


class DentalCavityInference:
    """
    Combined YOLO + UNet inference system
    
    Pipeline:
    1. YOLO detects cavity regions (bounding boxes)
    2. UNet segments cavity details within detected regions
    """
    
    def __init__(self,
                 yolo_weights: Optional[str] = None,
                 unet_weights: Optional[str] = None,
                 device: str = "cuda"):
        """
        Initialize inference system
        
        Args:
            yolo_weights: Path to trained YOLO weights
            unet_weights: Path to trained UNet weights
            device: Device to run inference on
        """
        self.device = device
        self.yolo_model = None
        self.unet_model = None
        
        if yolo_weights:
            self.load_yolo(yolo_weights)
        
        if unet_weights:
            self.load_unet(unet_weights)
    
    def load_yolo(self, weights_path: str):
        """Load YOLO model"""
        self.yolo_model = YOLODetector()
        self.yolo_model.load_weights(weights_path)
        logger.info(f"Loaded YOLO model from {weights_path}")
    
    def load_unet(self, weights_path: str):
        """Load UNet model"""
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.unet_model = UNet(in_channels=1, out_channels=1)
        self.unet_model.load_state_dict(checkpoint['model_state_dict'])
        self.unet_model = self.unet_model.to(self.device)
        self.unet_model.eval()
        logger.info(f"Loaded UNet model from {weights_path}")
    
    def preprocess_image(self, image_path: str, image_size: int = 640) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess X-ray image
        
        Args:
            image_path: Path to X-ray image
            image_size: Target size
        
        Returns:
            Preprocessed image and original image
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        original = image.copy()
        
        # Apply CLAHE enhancement
        image = apply_clahe(normalize_image(image))
        
        # Resize
        image = cv2.resize(image, (image_size, image_size))
        
        return image, original
    
    def detect_cavities_yolo(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect cavity regions using YOLO
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
        
        Returns:
            List of detections with bbox and confidence
        """
        if self.yolo_model is None:
            raise ValueError("YOLO model not loaded")
        
        detections = []
        results = self.yolo_model.predict(
            source=image,
            conf=conf_threshold
        )
        
        if results:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': cls
                    })
        
        return detections
    
    def segment_cavity_unet(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Segment cavity details within bounding box using UNet
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Segmentation mask
        """
        if self.unet_model is None:
            raise ValueError("UNet model not loaded")
        
        x1, y1, x2, y2 = bbox
        
        # Extract region
        region = image[y1:y2, x1:x2]
        
        # Preprocess region
        region = normalize_image(region)
        region = cv2.resize(region, (256, 256))
        
        # Convert to tensor
        region_tensor = torch.from_numpy(region).unsqueeze(0).unsqueeze(0).float()
        region_tensor = region_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            mask = self.unet_model(region_tensor)
        
        # Post-process
        mask = mask.cpu().numpy()[0, 0]
        mask = (mask > 0.5).astype(np.uint8)
        
        # Resize back to original size
        mask = cv2.resize(mask, (x2 - x1, y2 - y1))
        
        return mask
    
    def process_image(self, image_path: str, save_output: bool = True) -> Dict:
        """
        Process X-ray image with YOLO + UNet pipeline
        
        Args:
            image_path: Path to input image
            save_output: Whether to save visualizations
        
        Returns:
            Dictionary with results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Preprocess
        image, original = self.preprocess_image(image_path)
        
        # YOLO detection
        detections = self.detect_cavities_yolo(image)
        logger.info(f"Found {len(detections)} cavities")
        
        # UNet segmentation
        segmentations = []
        for i, detection in enumerate(detections):
            mask = self.segment_cavity_unet(image, detection['bbox'])
            segmentations.append({
                'detection_idx': i,
                'mask': mask
            })
        
        results = {
            'image_path': image_path,
            'detections': detections,
            'segmentations': segmentations,
            'num_cavities': len(detections)
        }
        
        if save_output:
            self._save_results(image_path, original, results)
        
        return results
    
    def _save_results(self, image_path: str, original: np.ndarray, results: Dict):
        """Save visualization results"""
        output_dir = Path("outputs/")
        output_dir.mkdir(exist_ok=True)
        
        image_name = Path(image_path).stem
        
        # Draw detections
        viz_image = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        for detection in results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(viz_image, f"{detection['confidence']:.2f}",
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save
        cv2.imwrite(str(output_dir / f"{image_name}_detections.jpg"), viz_image)
        logger.info(f"Saved detection visualization to {output_dir / f'{image_name}_detections.jpg'}")


def batch_inference(image_dir: str,
                   yolo_weights: str,
                   unet_weights: str,
                   device: str = "cuda"):
    """Process multiple images"""
    
    inference_system = DentalCavityInference(
        yolo_weights=yolo_weights,
        unet_weights=unet_weights,
        device=device
    )
    
    image_dir = Path(image_dir)
    results = []
    
    for image_path in image_dir.glob("*.jpg"):
        result = inference_system.process_image(str(image_path))
        results.append(result)
    
    return results
