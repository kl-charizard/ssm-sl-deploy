"""Custom CNN model implementations for sign language detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .base_model import BaseSignLanguageModel, ConvBlock, DepthwiseSeparableConv, SEBlock


class CustomCNN(BaseSignLanguageModel):
    """Custom CNN architecture optimized for sign language detection."""
    
    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.2,
        channels: List[int] = [64, 128, 256, 512],
        use_attention: bool = True
    ):
        """Initialize custom CNN.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            channels: List of channel sizes for each block
            use_attention: Whether to use attention mechanisms
        """
        super().__init__(num_classes, dropout_rate)
        
        self.channels = channels
        self.use_attention = use_attention
        
        # Feature extraction layers
        self.features = self._make_feature_layers()
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = self._make_classifier()
        
    def _make_feature_layers(self) -> nn.Sequential:
        """Create feature extraction layers."""
        layers = []
        in_channels = 3
        
        for i, out_channels in enumerate(self.channels):
            # Convolutional block
            layers.append(ConvBlock(in_channels, out_channels))
            layers.append(ConvBlock(out_channels, out_channels))
            
            # Add attention if specified
            if self.use_attention:
                layers.append(SEBlock(out_channels))
            
            # Max pooling
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            # Dropout for regularization
            if i > 0:  # Skip dropout for first block
                layers.append(nn.Dropout2d(self.dropout_rate * 0.5))
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _make_classifier(self) -> nn.Sequential:
        """Create classifier layers."""
        # Calculate the feature size after adaptive pooling
        feature_size = self.channels[-1] * 7 * 7
        
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class LightweightCNN(BaseSignLanguageModel):
    """Lightweight CNN model optimized for mobile deployment."""
    
    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.1,
        width_multiplier: float = 1.0,
        use_depthwise: bool = True
    ):
        """Initialize lightweight CNN.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            width_multiplier: Width multiplier for channels
            use_depthwise: Whether to use depthwise separable convolutions
        """
        super().__init__(num_classes, dropout_rate)
        
        self.width_multiplier = width_multiplier
        self.use_depthwise = use_depthwise
        
        # Calculate channels with width multiplier
        base_channels = [32, 64, 128, 256]
        self.channels = [int(ch * width_multiplier) for ch in base_channels]
        
        # Feature extraction
        self.features = self._make_feature_layers()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Linear(self.channels[-1], num_classes)
        
    def _make_feature_layers(self) -> nn.Sequential:
        """Create lightweight feature extraction layers."""
        layers = []
        in_channels = 3
        
        for i, out_channels in enumerate(self.channels):
            if self.use_depthwise and i > 0:
                # Use depthwise separable convolutions for efficiency
                layers.append(DepthwiseSeparableConv(
                    in_channels, out_channels, stride=2 if i == 0 else 1
                ))
            else:
                # Regular convolution for first layer
                stride = 2 if i > 0 else 1
                layers.append(ConvBlock(
                    in_channels, out_channels, stride=stride
                ))
            
            # Add pooling for spatial reduction
            if i < len(self.channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            # Minimal dropout
            if i > 1:
                layers.append(nn.Dropout2d(self.dropout_rate * 0.5))
                
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Feature extraction
        x = self.features(x)
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            downsample: Downsample module for skip connection
        """
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetLikeCNN(BaseSignLanguageModel):
    """ResNet-like architecture for sign language detection."""
    
    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.2,
        layers: List[int] = [2, 2, 2, 2],
        channels: List[int] = [64, 128, 256, 512]
    ):
        """Initialize ResNet-like CNN.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            layers: Number of residual blocks in each stage
            channels: Number of channels in each stage
        """
        super().__init__(num_classes, dropout_rate)
        
        self.channels = channels
        
        # Initial convolution
        self.conv1 = ConvBlock(3, channels[0], kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(channels[0], channels[0], layers[0])
        self.layer2 = self._make_layer(channels[0], channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual layer."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class EfficientBlock(nn.Module):
    """EfficientNet-inspired block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        se_ratio: float = 0.25
    ):
        """Initialize efficient block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            expand_ratio: Expansion ratio for inverted residual
            se_ratio: Squeeze-and-excitation ratio
        """
        super().__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.append(ConvBlock(in_channels, hidden_dim, kernel_size=1))
        
        # Depthwise
        layers.append(ConvBlock(
            hidden_dim, hidden_dim, kernel_size=kernel_size, 
            stride=stride, padding=kernel_size//2
        ))
        
        # Squeeze and excitation
        if se_ratio > 0:
            layers.append(SEBlock(hidden_dim, int(1 / se_ratio)))
        
        # Project
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        result = self.conv(x)
        
        if self.use_residual:
            result = result + x
        
        return result
