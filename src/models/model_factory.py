"""Model factory for creating sign language detection models."""

import torch
from typing import Dict, Any, Optional
from .base_model import BaseSignLanguageModel
from .cnn_models import CustomCNN, LightweightCNN, ResNetLikeCNN
from .pretrained_models import PretrainedModel, EnsembleModel
from ..utils.config import config


def create_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> BaseSignLanguageModel:
    """Factory function to create models.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for pretrained models)
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model instance
    """
    architecture = architecture.lower()
    
    # Get default dropout rate from config
    dropout_rate = kwargs.get('dropout_rate', config.get('model.dropout_rate', 0.2))
    
    # Custom CNN models
    if architecture == 'custom_cnn':
        return CustomCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            channels=kwargs.get('channels', [64, 128, 256, 512]),
            use_attention=kwargs.get('use_attention', True)
        )
    
    elif architecture == 'lightweight_cnn':
        return LightweightCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            width_multiplier=kwargs.get('width_multiplier', 1.0),
            use_depthwise=kwargs.get('use_depthwise', True)
        )
    
    elif architecture == 'resnet_like':
        return ResNetLikeCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            layers=kwargs.get('layers', [2, 2, 2, 2]),
            channels=kwargs.get('channels', [64, 128, 256, 512])
        )
    
    # Pretrained models
    elif architecture in [
        'resnet18', 'resnet50', 'resnet101',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
        'densenet121', 'vgg16', 'vgg19'
    ]:
        return PretrainedModel(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            fine_tune=kwargs.get('fine_tune', True),
            freeze_layers=kwargs.get('freeze_layers', None)
        )
    
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def create_ensemble_model(
    architectures: list,
    num_classes: int,
    voting_method: str = 'soft',
    **kwargs
) -> EnsembleModel:
    """Create an ensemble of models.
    
    Args:
        architectures: List of architecture names
        num_classes: Number of output classes
        voting_method: 'soft' or 'hard' voting
        **kwargs: Additional arguments for individual models
        
    Returns:
        Ensemble model instance
    """
    models = []
    
    for arch in architectures:
        model = create_model(
            architecture=arch,
            num_classes=num_classes,
            **kwargs
        )
        models.append(model)
    
    return EnsembleModel(
        models=models,
        num_classes=num_classes,
        voting_method=voting_method
    )


def get_model_recommendations(
    target_platform: str = 'desktop',
    performance_priority: str = 'balanced'  # 'speed', 'accuracy', 'balanced'
) -> Dict[str, Any]:
    """Get model recommendations based on target platform and priorities.
    
    Args:
        target_platform: 'desktop', 'mobile', 'embedded'
        performance_priority: 'speed', 'accuracy', 'balanced'
        
    Returns:
        Dictionary with recommended architectures and settings
    """
    recommendations = {}
    
    if target_platform == 'desktop':
        if performance_priority == 'accuracy':
            recommendations = {
                'primary': 'efficientnet_b2',
                'alternatives': ['resnet50', 'densenet121'],
                'ensemble': ['efficientnet_b1', 'resnet50', 'mobilenet_v3_large'],
                'batch_size': 32,
                'input_size': [224, 224]
            }
        elif performance_priority == 'speed':
            recommendations = {
                'primary': 'mobilenet_v3_large',
                'alternatives': ['efficientnet_b0', 'resnet18'],
                'ensemble': ['mobilenet_v3_large', 'efficientnet_b0'],
                'batch_size': 64,
                'input_size': [224, 224]
            }
        else:  # balanced
            recommendations = {
                'primary': 'efficientnet_b0',
                'alternatives': ['mobilenet_v3_large', 'resnet18'],
                'ensemble': ['efficientnet_b0', 'mobilenet_v3_large'],
                'batch_size': 32,
                'input_size': [224, 224]
            }
    
    elif target_platform == 'mobile':
        if performance_priority == 'accuracy':
            recommendations = {
                'primary': 'efficientnet_b0',
                'alternatives': ['mobilenet_v3_large', 'lightweight_cnn'],
                'ensemble': None,  # Ensembles not recommended for mobile
                'batch_size': 16,
                'input_size': [224, 224],
                'quantization': True
            }
        else:  # speed or balanced
            recommendations = {
                'primary': 'lightweight_cnn',
                'alternatives': ['mobilenet_v3_small', 'mobilenet_v2'],
                'ensemble': None,
                'batch_size': 16,
                'input_size': [160, 160],  # Smaller input for mobile
                'quantization': True,
                'width_multiplier': 0.75 if performance_priority == 'speed' else 1.0
            }
    
    elif target_platform == 'embedded':
        recommendations = {
            'primary': 'lightweight_cnn',
            'alternatives': ['mobilenet_v3_small'],
            'ensemble': None,
            'batch_size': 8,
            'input_size': [128, 128],  # Very small input for embedded
            'quantization': True,
            'width_multiplier': 0.5,
            'use_depthwise': True
        }
    
    else:
        raise ValueError(f"Unsupported target platform: {target_platform}")
    
    return recommendations


def load_model_from_config(
    model_config: Dict[str, Any],
    num_classes: int,
    checkpoint_path: Optional[str] = None
) -> BaseSignLanguageModel:
    """Load model from configuration dictionary.
    
    Args:
        model_config: Model configuration dictionary
        num_classes: Number of output classes
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Configured model instance
    """
    architecture = model_config.get('architecture', 'efficientnet_b0')
    
    # Create model
    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=model_config.get('pretrained', True),
        dropout_rate=model_config.get('dropout_rate', 0.2),
        **model_config.get('model_kwargs', {})
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {checkpoint_path}")
    
    return model


def get_model_complexity_analysis(model: BaseSignLanguageModel) -> Dict[str, Any]:
    """Analyze model complexity and performance characteristics.
    
    Args:
        model: Model instance to analyze
        
    Returns:
        Dictionary with complexity analysis
    """
    import torch
    
    model_info = model.get_model_info()
    
    # Estimate FLOPs (simplified)
    def estimate_flops(model, input_size=(1, 3, 224, 224)):
        """Rough FLOP estimation."""
        try:
            dummy_input = torch.randn(input_size)
            model.eval()
            
            # Count parameters as a proxy for complexity
            total_params = model_info['total_parameters']
            
            # Rough estimation: assume each parameter is used once per forward pass
            # This is a very rough approximation
            estimated_flops = total_params * input_size[0]  # multiply by batch size
            
            return estimated_flops
        except Exception:
            return None
    
    flops = estimate_flops(model)
    
    # Categorize model size
    params = model_info['total_parameters']
    if params < 1e6:
        size_category = 'tiny'
    elif params < 5e6:
        size_category = 'small'
    elif params < 20e6:
        size_category = 'medium'
    elif params < 100e6:
        size_category = 'large'
    else:
        size_category = 'very_large'
    
    # Estimate inference time category (rough)
    if 'lightweight' in str(type(model)).lower() or 'mobilenet' in str(type(model)).lower():
        speed_category = 'fast'
    elif 'efficientnet_b0' in str(type(model)).lower() or 'resnet18' in str(type(model)).lower():
        speed_category = 'medium'
    else:
        speed_category = 'slow'
    
    analysis = {
        **model_info,
        'estimated_flops': flops,
        'size_category': size_category,
        'speed_category': speed_category,
        'mobile_friendly': params < 10e6,
        'embedded_friendly': params < 2e6,
        'memory_usage_mb': params * 4 / (1024 * 1024)  # Assuming float32
    }
    
    return analysis


# Model registry for easy access
MODEL_REGISTRY = {
    'custom_cnn': CustomCNN,
    'lightweight_cnn': LightweightCNN,
    'resnet_like': ResNetLikeCNN,
    'pretrained': PretrainedModel,
    'ensemble': EnsembleModel
}
