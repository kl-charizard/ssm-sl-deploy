"""Optimizer and scheduler utilities for training."""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau,
    CosineAnnealingWarmRestarts, OneCycleLR
)
import math
from typing import Dict, Any, Optional


def get_optimizer(
    model_parameters,
    optimizer_name: str = 'adam',
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """Get optimizer by name.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_name: Name of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay factor
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_name == 'adamw':
        return optim.AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_name == 'sgd':
        return optim.SGD(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            alpha=kwargs.get('alpha', 0.99)
        )
    
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'cosine',
    total_epochs: int = 100,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Get learning rate scheduler by name.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler
        total_epochs: Total number of training epochs
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'step':
        step_size = kwargs.get('step_size', total_epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_name == 'cosine_warm_restart':
        T_0 = kwargs.get('T_0', 10)
        T_mult = kwargs.get('T_mult', 2)
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_name == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            min_lr=kwargs.get('min_lr', 1e-6),
            verbose=True
        )
    
    elif scheduler_name == 'one_cycle':
        max_lr = kwargs.get('max_lr', optimizer.defaults['lr'] * 10)
        steps_per_epoch = kwargs.get('steps_per_epoch', 100)
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=kwargs.get('pct_start', 0.3),
            div_factor=kwargs.get('div_factor', 25),
            final_div_factor=kwargs.get('final_div_factor', 10000)
        )
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


class WarmupScheduler:
    """Learning rate warmup scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_epochs: Number of warmup epochs
            base_scheduler: Base scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1
        
        if self.epoch <= self.warmup_epochs:
            # Warmup phase
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base_lr * self.epoch / self.warmup_epochs
        elif self.base_scheduler is not None:
            # Use base scheduler after warmup
            self.base_scheduler.step()
    
    def state_dict(self):
        """Get scheduler state dict."""
        state = {'epoch': self.epoch, 'base_lrs': self.base_lrs}
        if self.base_scheduler is not None:
            state['base_scheduler'] = self.base_scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        """Load scheduler state dict."""
        self.epoch = state_dict['epoch']
        self.base_lrs = state_dict['base_lrs']
        if self.base_scheduler is not None and 'base_scheduler' in state_dict:
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


def create_optimizer_with_schedule(
    model,
    optimizer_config: Dict[str, Any],
    scheduler_config: Dict[str, Any],
    total_epochs: int,
    steps_per_epoch: int = 100
) -> tuple:
    """Create optimizer and scheduler from configurations.
    
    Args:
        model: Model to optimize
        optimizer_config: Optimizer configuration
        scheduler_config: Scheduler configuration
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create optimizer
    optimizer = get_optimizer(
        model.parameters(),
        optimizer_name=optimizer_config.get('name', 'adam'),
        learning_rate=optimizer_config.get('lr', 0.001),
        weight_decay=optimizer_config.get('weight_decay', 1e-4),
        **optimizer_config.get('params', {})
    )
    
    # Create base scheduler
    base_scheduler = None
    if scheduler_config.get('name') is not None:
        scheduler_params = scheduler_config.get('params', {})
        scheduler_params['steps_per_epoch'] = steps_per_epoch
        
        base_scheduler = get_scheduler(
            optimizer,
            scheduler_name=scheduler_config['name'],
            total_epochs=total_epochs,
            **scheduler_params
        )
    
    # Add warmup if specified
    warmup_epochs = scheduler_config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            base_scheduler=base_scheduler
        )
    else:
        scheduler = base_scheduler
    
    return optimizer, scheduler


def get_parameter_groups(
    model,
    weight_decay: float = 1e-4,
    no_decay_keywords: list = None
) -> list:
    """Create parameter groups with different weight decay settings.
    
    Args:
        model: Model to create parameter groups for
        weight_decay: Default weight decay
        no_decay_keywords: Keywords for parameters that shouldn't have weight decay
        
    Returns:
        List of parameter groups
    """
    if no_decay_keywords is None:
        no_decay_keywords = ['bias', 'norm', 'bn']
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should not have weight decay
        if any(keyword in name.lower() for keyword in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    return param_groups


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay: float = 0.999):
        """Initialize EMA.
        
        Args:
            model: Model to track
            decay: Decay factor for EMA
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def cosine_annealing_with_warmup(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 5,
    base_lr: float = 0.001,
    min_lr: float = 1e-6
) -> float:
    """Calculate learning rate with cosine annealing and warmup.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate for current epoch
    """
    if epoch < warmup_epochs:
        # Warmup phase
        return base_lr * epoch / warmup_epochs
    else:
        # Cosine annealing
        cos_epoch = epoch - warmup_epochs
        cos_total = total_epochs - warmup_epochs
        return min_lr + (base_lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * cos_epoch / cos_total)
        )
