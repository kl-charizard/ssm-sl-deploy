"""Base model class for sign language detection."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch.nn.functional as F


class BaseSignLanguageModel(nn.Module, ABC):
    """Abstract base class for sign language detection models."""
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.2):
        """Initialize base model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions with the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def freeze_layers(self, layer_names: Optional[list] = None):
        """Freeze specific layers or all layers.
        
        Args:
            layer_names: List of layer names to freeze. If None, freezes all layers.
        """
        if layer_names is None:
            # Freeze all layers
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
    
    def unfreeze_layers(self, layer_names: Optional[list] = None):
        """Unfreeze specific layers or all layers.
        
        Args:
            layer_names: List of layer names to unfreeze. If None, unfreezes all layers.
        """
        if layer_names is None:
            # Unfreeze all layers
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None, 
                       scheduler_state: Optional[Dict] = None, **kwargs):
        """Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            **kwargs: Additional information to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            **kwargs
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath: str, num_classes: int, **model_kwargs) -> Tuple['BaseSignLanguageModel', Dict]:
        """Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            num_classes: Number of classes for model initialization
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            Tuple of (loaded_model, checkpoint_info)
        """
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Create model instance
        model = cls(num_classes=num_classes, **model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract additional info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'model_info': checkpoint.get('model_info', {}),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'scheduler_state_dict': checkpoint.get('scheduler_state_dict')
        }
        
        return model, checkpoint_info


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = 'relu'
    ):
        """Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            use_bn: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'gelu')
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficient mobile models."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        """Initialize depthwise separable convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
        """
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        """Initialize SE block.
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio
        """
        super().__init__()
        
        reduced_channels = max(1, channels // reduction)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.squeeze(x).view(b, c)
        
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        
        # Scale
        return x * y
