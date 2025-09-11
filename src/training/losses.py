"""Loss functions for sign language detection training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Prediction logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        """Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Prediction logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Label smoothed cross entropy loss
        """
        log_prob = F.log_softmax(inputs, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross entropy loss for handling class imbalance."""
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """Initialize weighted cross entropy loss.
        
        Args:
            class_weights: Weights for each class
        """
        super().__init__()
        self.register_buffer('class_weights', class_weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Prediction logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Weighted cross entropy loss
        """
        return F.cross_entropy(inputs, targets, weight=self.class_weights)


class MixupLoss(nn.Module):
    """Mixup loss for data augmentation at the loss level."""
    
    def __init__(self, criterion: nn.Module):
        """Initialize mixup loss.
        
        Args:
            criterion: Base loss function
        """
        super().__init__()
        self.criterion = criterion
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """Forward pass for mixup loss.
        
        Args:
            inputs: Prediction logits
            targets_a: First set of targets
            targets_b: Second set of targets
            lam: Mixup lambda parameter
            
        Returns:
            Mixup loss value
        """
        loss_a = self.criterion(inputs, targets_a)
        loss_b = self.criterion(inputs, targets_b)
        return lam * loss_a + (1 - lam) * loss_b


def get_loss_function(
    loss_type: str = 'cross_entropy',
    num_classes: int = 29,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """Get loss function by name.
    
    Args:
        loss_type: Type of loss function
        num_classes: Number of classes
        class_weights: Class weights for weighted loss
        **kwargs: Additional loss-specific arguments
        
    Returns:
        Loss function instance
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'weighted_cross_entropy':
        return WeightedCrossEntropyLoss(class_weights=class_weights)
    
    elif loss_type == 'focal_loss':
        alpha = kwargs.get('focal_alpha', 1.0)
        gamma = kwargs.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('label_smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def calculate_class_weights(
    class_counts: Dict[int, int],
    num_classes: int,
    method: str = 'inverse_freq'
) -> torch.Tensor:
    """Calculate class weights for handling imbalanced datasets.
    
    Args:
        class_counts: Dictionary mapping class indices to counts
        num_classes: Total number of classes
        method: Method for calculating weights ('inverse_freq', 'effective_num')
        
    Returns:
        Class weights tensor
    """
    total_samples = sum(class_counts.values())
    weights = torch.ones(num_classes)
    
    if method == 'inverse_freq':
        for class_idx, count in class_counts.items():
            if count > 0:
                weights[class_idx] = total_samples / (num_classes * count)
    
    elif method == 'effective_num':
        # Effective number of samples method
        beta = 0.9999  # Hyperparameter
        effective_num = 1.0 - torch.pow(beta, torch.tensor(list(class_counts.values())))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights


class DistillationLoss(nn.Module):
    """Knowledge distillation loss."""
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        student_loss_fn: nn.Module = None
    ):
        """Initialize distillation loss.
        
        Args:
            temperature: Temperature for softmax
            alpha: Weight for distillation loss vs student loss
            student_loss_fn: Loss function for student model
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.student_loss_fn = student_loss_fn or nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            labels: Ground truth labels
            
        Returns:
            Combined distillation loss
        """
        # Distillation loss
        student_log_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_prob = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = self.kl_loss(student_log_prob, teacher_prob) * (self.temperature ** 2)
        
        # Student loss
        student_loss = self.student_loss_fn(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        
        return total_loss


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple:
    """Apply mixup augmentation to data.
    
    Args:
        x: Input batch (batch_size, channels, height, width)
        y: Labels (batch_size,)
        alpha: Mixup alpha parameter
        
    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda)
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple:
    """Apply CutMix augmentation to data.
    
    Args:
        x: Input batch (batch_size, channels, height, width)
        y: Labels (batch_size,)
        alpha: CutMix alpha parameter
        
    Returns:
        Tuple of (mixed_x, y_a, y_b, lambda)
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    _, _, H, W = x.shape
    
    # Generate random bounding box
    cut_rat = torch.sqrt(1. - lam)
    cut_w = (W * cut_rat).int()
    cut_h = (H * cut_rat).int()
    
    # Random center coordinates
    cx = torch.randint(0, W, (1,))
    cy = torch.randint(0, H, (1,))
    
    # Bounding box coordinates
    bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
    bby1 = torch.clamp(cy - cut_h // 2, 0, H)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
    bby2 = torch.clamp(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match the actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam
