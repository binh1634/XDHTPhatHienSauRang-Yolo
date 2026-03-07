"""
YOLO Model for dental cavity detection
"""

from ultralytics import YOLO
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO-based dental cavity detector"""
    
    def __init__(self, model_name: str = "yolov8m", pretrained: bool = True):
        """
        Initialize YOLO model
        
        Args:
            model_name: YOLO model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            pretrained: Whether to use pretrained weights
        """
        self.model_name = model_name
        self.model = YOLO(f"{model_name}.pt")
        self.device = None
        logger.info(f"Initialized YOLO model: {model_name}")
    
    def train(self, 
              data_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              learning_rate: float = 0.001,
              patience: int = 20,
              device: str = "0",
              save_dir: str = "runs/detect"):
        """
        Train YOLO model
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience
            device: Device to train on (GPU index or "cpu")
            save_dir: Directory to save results
        """
        logger.info("Starting YOLO training...")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            lr0=learning_rate,
            patience=patience,
            device=device,
            project=save_dir,
            name="cavity_detection",
            save=True,
            verbose=True,
            plots=True
        )
        
        logger.info(f"Training completed. Results saved to {save_dir}")
        return results
    
    def validate(self, data_yaml: str, device: str = "0"):
        """Validate YOLO model"""
        logger.info("Starting validation...")
        metrics = self.model.val(data=data_yaml, device=device)
        return metrics
    
    def predict(self, source, conf: float = 0.5, iou: float = 0.45):
        """
        Run inference on images
        
        Args:
            source: Image path, URL, or directory
            conf: Confidence threshold
            iou: IoU threshold for NMS
        
        Returns:
            Predictions object
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            verbose=False
        )
        return results
    
    def load_weights(self, weights_path: str):
        """Load trained weights"""
        self.model = YOLO(weights_path)
        logger.info(f"Loaded weights from {weights_path}")
    
    def save_weights(self, output_path: str):
        """Save model weights"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(output_path)
        logger.info(f"Saved weights to {output_path}")
    
    def export(self, format: str = "onnx", output_path: str = "models/yolo_cavity.onnx"):
        """
        Export model to different format
        
        Args:
            format: Export format (onnx, torchscript, tflite, etc.)
            output_path: Output file path
        """
        path = self.model.export(format=format)
        logger.info(f"Model exported to {format} format: {path}")
        return path
