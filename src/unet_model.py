"""
UNet Model for dental cavity segmentation
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block"""
    
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Pad if necessary
        if x.shape != skip.shape:
            x = self._pad_to_match(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
    @staticmethod
    def _pad_to_match(x, skip):
        """Pad x to match skip connection dimensions"""
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = nn.functional.pad(x, (diff_w // 2, diff_w - diff_w // 2,
                                   diff_h // 2, diff_h - diff_h // 2))
        return x


class UNet(nn.Module):
    """
    UNet architecture for medical image segmentation
    
    Suitable for segmenting dental cavities in X-ray images
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: list = None):
        """
        Initialize UNet model
        
        Args:
            in_channels: Number of input channels (1 for grayscale X-ray)
            out_channels: Number of output channels (1 for binary segmentation)
            features: Number of features at each level
        """
        super(UNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.features = features
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])
        
        # Encoder (downsampling path)
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # Bottleneck
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        
        # Decoder (upsampling path)
        self.up1 = Up(features[3] * 2 + features[3], features[3])
        self.up2 = Up(features[3] + features[2], features[2])
        self.up3 = Up(features[2] + features[1], features[1])
        self.up4 = Up(features[1] + features[0], features[0])
        
        # Output convolution
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.pool(x4)
        x5 = self.bottleneck(x5)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        output = self.sigmoid(logits)
        
        return output


class AttentionUNet(nn.Module):
    """
    UNet with Attention Gates for better feature extraction
    
    Attention gates help the model focus on relevant regions
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: list = None):
        super(AttentionUNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        # Similar to UNet but with attention mechanisms
        # For simplicity, we'll use the basic UNet
        # In a full implementation, add attention gates here
        self.unet = UNet(in_channels, out_channels, features)
    
    def forward(self, x):
        return self.unet(x)


def create_unet_model(in_channels: int = 1,
                     out_channels: int = 1,
                     features: list = None,
                     device: str = "cuda") -> UNet:
    """
    Create and return a UNet model
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        features: Feature channels at each level
        device: Device to put model on
    
    Returns:
        UNet model on specified device
    """
    model = UNet(in_channels, out_channels, features)
    model = model.to(device)
    logger.info(f"Created UNet model on device: {device}")
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
