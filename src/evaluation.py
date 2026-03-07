"""
Evaluation module for model performance assessment
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Metrics for segmentation evaluation"""
    
    @staticmethod
    def dice_coefficient(predictions: np.ndarray, targets: np.ndarray, smooth: float = 1.0) -> float:
        """
        Dice Coefficient = 2|X∩Y|/(|X|+|Y|)
        
        Range: 0-1, where 1 is perfect overlap
        """
        intersection = np.logical_and(predictions, targets).sum()
        return (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    @staticmethod
    def iou(predictions: np.ndarray, targets: np.ndarray, smooth: float = 1.0) -> float:
        """
        Intersection over Union (IoU) = |X∩Y|/|X∪Y|
        
        Range: 0-1, where 1 is perfect overlap
        """
        intersection = np.logical_and(predictions, targets).sum()
        union = np.logical_or(predictions, targets).sum()
        return (intersection + smooth) / (union + smooth)
    
    @staticmethod
    def sensitivity(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Sensitivity (True Positive Rate) = TP / (TP + FN)"""
        tp = np.logical_and(predictions, targets).sum()
        fn = np.logical_and(~predictions, targets).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    @staticmethod
    def specificity(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Specificity (True Negative Rate) = TN / (TN + FP)"""
        tn = np.logical_and(~predictions, ~targets).sum()
        fp = np.logical_and(predictions, ~targets).sum()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    @staticmethod
    def hausdorff_distance(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Hausdorff Distance: maximum distance from a point in predictions to nearest point in targets
        
        Lower is better
        """
        from scipy.spatial.distance import cdist
        
        pred_points = np.argwhere(predictions)
        target_points = np.argwhere(targets)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        distances = cdist(pred_points, target_points)
        return max(distances.min(axis=1).max(), distances.min(axis=0).max())


class DetectionMetrics:
    """Metrics for object detection evaluation"""
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes in format [x1, y1, x2, y2]"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
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
    
    @staticmethod
    def average_precision(predictions: List[Dict],
                         ground_truths: List[Dict],
                         iou_threshold: float = 0.5) -> float:
        """
        Calculate Average Precision (AP)
        
        Args:
            predictions: List of predictions with 'bbox' and 'confidence'
            ground_truths: List of ground truth with 'bbox'
            iou_threshold: IoU threshold for match
        
        Returns:
            Average Precision score (0-1)
        """
        if len(predictions) == 0:
            return 0.0
        
        # Sort predictions by confidence
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        matched_gt = set()
        
        for i, pred in enumerate(predictions):
            max_iou = 0
            max_idx = -1
            
            for j, gt in enumerate(ground_truths):
                if j in matched_gt:
                    continue
                
                iou = DetectionMetrics.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= iou_threshold and max_idx >= 0:
                tp[i] = 1
                matched_gt.add(max_idx)
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP using all points interpolation
        ap = 0.0
        for i in range(len(precisions)):
            ap += precisions[i] * (recalls[i] - (recalls[i-1] if i > 0 else 0))
        
        return ap
    
    @staticmethod
    def mean_average_precision(predictions: List[List[Dict]],
                              ground_truths: List[List[Dict]],
                              iou_threshold: float = 0.5) -> float:
        """Calculate mAP across multiple images"""
        aps = []
        
        for pred, gt in zip(predictions, ground_truths):
            ap = DetectionMetrics.average_precision(pred, gt, iou_threshold)
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0


class Evaluator:
    """Unified evaluator for both detection and segmentation"""
    
    def __init__(self, task: str = "both"):
        """
        Args:
            task: 'detection', 'segmentation', or 'both'
        """
        self.task = task
        self.seg_metrics = SegmentationMetrics()
        self.det_metrics = DetectionMetrics()
    
    def evaluate_segmentation(self,
                             predictions: List[np.ndarray],
                             targets: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate segmentation results"""
        results = {
            'dice': [],
            'iou': [],
            'sensitivity': [],
            'specificity': []
        }
        
        for pred, target in zip(predictions, targets):
            # Binary threshold
            pred_binary = (pred > 0.5).astype(np.uint8)
            target_binary = (target > 0.5).astype(np.uint8)
            
            results['dice'].append(
                self.seg_metrics.dice_coefficient(pred_binary, target_binary)
            )
            results['iou'].append(
                self.seg_metrics.iou(pred_binary, target_binary)
            )
            results['sensitivity'].append(
                self.seg_metrics.sensitivity(pred_binary, target_binary)
            )
            results['specificity'].append(
                self.seg_metrics.specificity(pred_binary, target_binary)
            )
        
        # Average metrics
        avg_results = {
            'mean_dice': np.mean(results['dice']),
            'mean_iou': np.mean(results['iou']),
            'mean_sensitivity': np.mean(results['sensitivity']),
            'mean_specificity': np.mean(results['specificity']),
            'std_dice': np.std(results['dice']),
            'std_iou': np.std(results['iou'])
        }
        
        return avg_results
    
    def evaluate_detection(self,
                          predictions: List[List[Dict]],
                          ground_truths: List[List[Dict]],
                          iou_thresholds: List[float] = None) -> Dict[str, float]:
        """Evaluate detection results"""
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75]
        
        results = {}
        
        for threshold in iou_thresholds:
            ap = self.det_metrics.mean_average_precision(
                predictions, ground_truths, threshold
            )
            results[f'ap_{threshold}'] = ap
        
        # Overall mAP
        results['mAP'] = np.mean([results[f'ap_{t}'] for t in iou_thresholds])
        
        return results
    
    def print_report(self, metrics: Dict[str, float]):
        """Print evaluation report"""
        logger.info("\n" + "="*50)
        logger.info("EVALUATION REPORT")
        logger.info("="*50)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key:.<40} {value:.4f}")
        
        logger.info("="*50)
