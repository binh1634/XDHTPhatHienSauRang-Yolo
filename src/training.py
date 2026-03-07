"""
Training pipeline for dental cavity detection models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple, Dict, Optional

from src.unet_model import UNet, create_unet_model
from src.data_loader import create_data_loaders
from src.utils import load_config

logger = logging.getLogger(__name__)


class DiceCoefficient:
    """Dice coefficient metric for segmentation"""
    
    @staticmethod
    def forward(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> float:
        """
        Calculate Dice coefficient
        
        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Ground truth (B, C, H, W)
            smooth: Smoothing constant
        
        Returns:
            Dice score
        """
        intersection = (predictions * targets).sum()
        return (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)


class SegmentationTrainer:
    """Trainer for UNet segmentation model"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0005):
        """
        Initialize trainer
        
        Args:
            model: UNet model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        self.dice_metric = DiceCoefficient()
        self.best_dice = 0.0
        
        logger.info(f"Initialized SegmentationTrainer on device: {device}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                dice = self.dice_metric.forward(outputs, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'dice': dice.item()
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        
        return {
            'train_loss': avg_loss,
            'train_dice': avg_dice
        }
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating")
            
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice = self.dice_metric.forward(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice.item()
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'dice': dice.item()
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        return {
            'val_loss': avg_loss,
            'val_dice': avg_dice
        }
    
    def train(self,
              train_loader,
              val_loader,
              epochs: int = 100,
              checkpoint_dir: str = "models/",
              patience: int = 20):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints
            patience: Early stopping patience
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir="logs/")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Log
            for key, value in train_metrics.items():
                writer.add_scalar(f'Training/{key}', value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f'Validation/{key}', value, epoch)
            
            self.scheduler.step()
            
            # Early stopping
            current_dice = val_metrics['val_dice']
            if current_dice > self.best_dice:
                self.best_dice = current_dice
                patience_counter = 0
                
                # Save best model
                checkpoint_path = Path(checkpoint_dir) / "best_unet.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': self.best_dice
                }, checkpoint_path)
                
                logger.info(f"Saved best model with Dice: {self.best_dice:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Train Dice: {train_metrics['train_dice']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                       f"Val Dice: {val_metrics['val_dice']:.4f}")
        
        writer.close()
        logger.info("Training completed!")


def train_unet_from_config(config_path: str = "config.yaml"):
    """Train UNet model using config file"""
    
    # Load config
    config = load_config(config_path)
    
    # Create model
    unet_config = config['unet']
    model = create_unet_model(
        in_channels=unet_config['in_channels'],
        out_channels=unet_config['out_channels'],
        features=unet_config['features'],
        device=config['training']['device']
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=config['paths']['processed_data'],
        batch_size=config['training']['batch_size'],
        image_size=config['dataset']['image_size'],
        task="segmentation",
        num_workers=config['training']['num_workers']
    )
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        device=config['training']['device'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        checkpoint_dir=config['paths']['models_dir'],
        patience=20
    )


if __name__ == "__main__":
    train_unet_from_config()
