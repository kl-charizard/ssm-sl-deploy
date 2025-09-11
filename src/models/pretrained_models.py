"""Pretrained model implementations for sign language detection."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
from .base_model import BaseSignLanguageModel


class PretrainedModel(BaseSignLanguageModel):
    """Wrapper for pretrained models."""
    
    def __init__(
        self,
        architecture: str,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        fine_tune: bool = True,
        freeze_layers: Optional[int] = None
    ):
        """Initialize pretrained model.
        
        Args:
            architecture: Model architecture name
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for classifier
            fine_tune: Whether to fine-tune the backbone
            freeze_layers: Number of initial layers to freeze (if fine_tune=True)
        """
        super().__init__(num_classes, dropout_rate)
        
        self.architecture = architecture.lower()
        self.pretrained = pretrained
        self.fine_tune = fine_tune
        
        # Load backbone
        self.backbone = self._load_backbone()
        
        # Modify classifier
        self._modify_classifier()
        
        # Handle layer freezing
        if freeze_layers is not None and freeze_layers > 0:
            self._freeze_initial_layers(freeze_layers)
        elif not fine_tune:
            self._freeze_backbone()
    
    def _load_backbone(self) -> nn.Module:
        """Load the backbone model."""
        if self.architecture == 'resnet18':
            return models.resnet18(pretrained=self.pretrained)
        elif self.architecture == 'resnet50':
            return models.resnet50(pretrained=self.pretrained)
        elif self.architecture == 'resnet101':
            return models.resnet101(pretrained=self.pretrained)
        elif self.architecture == 'efficientnet_b0':
            return models.efficientnet_b0(pretrained=self.pretrained)
        elif self.architecture == 'efficientnet_b1':
            return models.efficientnet_b1(pretrained=self.pretrained)
        elif self.architecture == 'efficientnet_b2':
            return models.efficientnet_b2(pretrained=self.pretrained)
        elif self.architecture == 'mobilenet_v2':
            return models.mobilenet_v2(pretrained=self.pretrained)
        elif self.architecture == 'mobilenet_v3_small':
            return models.mobilenet_v3_small(pretrained=self.pretrained)
        elif self.architecture == 'mobilenet_v3_large':
            return models.mobilenet_v3_large(pretrained=self.pretrained)
        elif self.architecture == 'densenet121':
            return models.densenet121(pretrained=self.pretrained)
        elif self.architecture == 'vgg16':
            return models.vgg16(pretrained=self.pretrained)
        elif self.architecture == 'vgg19':
            return models.vgg19(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
    
    def _modify_classifier(self):
        """Modify the classifier head for sign language detection."""
        if 'resnet' in self.architecture:
            # ResNet models
            num_features = self.backbone.fc.in_features
            self.backbone.fc = self._create_classifier(num_features)
            
        elif 'efficientnet' in self.architecture:
            # EfficientNet models
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._create_classifier(num_features)
            
        elif 'mobilenet' in self.architecture:
            # MobileNet models
            if hasattr(self.backbone.classifier, '__getitem__'):
                # MobileNetV3
                num_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(num_features, self.num_classes)
            else:
                # MobileNetV2
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = self._create_classifier(num_features)
                
        elif 'densenet' in self.architecture:
            # DenseNet models
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = self._create_classifier(num_features)
            
        elif 'vgg' in self.architecture:
            # VGG models
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_features, self.num_classes)
            
        else:
            raise ValueError(f"Classifier modification not implemented for {self.architecture}")
    
    def _create_classifier(self, num_features: int) -> nn.Module:
        """Create a custom classifier head.
        
        Args:
            num_features: Number of input features
            
        Returns:
            Classifier module
        """
        if num_features > 1024:
            # For large feature sizes, use a more complex classifier
            return nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate * 0.5),
                nn.Linear(256, self.num_classes)
            )
        elif num_features > 512:
            # Medium classifier
            return nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate * 0.5),
                nn.Linear(256, self.num_classes)
            )
        else:
            # Simple classifier for smaller models
            return nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(num_features, self.num_classes)
            )
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        if 'resnet' in self.architecture:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif 'efficientnet' in self.architecture:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif 'mobilenet' in self.architecture:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif 'densenet' in self.architecture:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif 'vgg' in self.architecture:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
    
    def _freeze_initial_layers(self, num_layers: int):
        """Freeze initial layers of the backbone.
        
        Args:
            num_layers: Number of layers to freeze
        """
        layer_count = 0
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                if layer_count < num_layers:
                    param.requires_grad = False
                    layer_count += 1
                else:
                    break
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor:
        """Get intermediate feature maps.
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract features from
            
        Returns:
            Feature maps from specified layer
        """
        features = []
        
        def hook(module, input, output):
            features.append(output)
        
        # Register hook
        if layer_name:
            layer = dict(self.backbone.named_modules())[layer_name]
            handle = layer.register_forward_hook(hook)
        else:
            # Use the last convolutional layer before classifier
            if 'resnet' in self.architecture:
                handle = self.backbone.avgpool.register_forward_hook(hook)
            elif 'efficientnet' in self.architecture:
                handle = self.backbone.avgpool.register_forward_hook(hook)
            elif 'mobilenet' in self.architecture:
                handle = self.backbone.features[-1].register_forward_hook(hook)
            else:
                raise NotImplementedError(f"Feature extraction not implemented for {self.architecture}")
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hook
        handle.remove()
        
        return features[0] if features else None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get extended model information."""
        base_info = super().get_model_info()
        base_info.update({
            'architecture': self.architecture,
            'pretrained': self.pretrained,
            'fine_tune': self.fine_tune,
        })
        return base_info


class EnsembleModel(BaseSignLanguageModel):
    """Ensemble of multiple pretrained models."""
    
    def __init__(
        self,
        models: list,
        num_classes: int,
        voting_method: str = 'soft'
    ):
        """Initialize ensemble model.
        
        Args:
            models: List of model instances
            num_classes: Number of output classes
            voting_method: 'soft' (average probabilities) or 'hard' (majority vote)
        """
        super().__init__(num_classes, 0.0)  # No dropout for ensemble
        
        self.models = nn.ModuleList(models)
        self.voting_method = voting_method
        self.num_models = len(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        if self.voting_method == 'soft':
            # Average the outputs
            ensemble_output = torch.stack(outputs).mean(dim=0)
        else:
            # Hard voting - take mode of predictions
            predictions = [torch.argmax(output, dim=1) for output in outputs]
            # This is a simplified hard voting - for full implementation,
            # we'd need to handle ties properly
            ensemble_output = outputs[0]  # Placeholder
        
        return ensemble_output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make ensemble predictions."""
        self.eval()
        
        with torch.no_grad():
            if self.voting_method == 'soft':
                logits = self.forward(x)
                predictions = torch.argmax(logits, dim=1)
            else:
                # Hard voting
                all_predictions = []
                for model in self.models:
                    model_predictions = model.predict(x)
                    all_predictions.append(model_predictions)
                
                # Take majority vote
                stacked_predictions = torch.stack(all_predictions, dim=0)
                predictions = torch.mode(stacked_predictions, dim=0)[0]
        
        return predictions
