"""Main trainer class for sign language detection models."""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb

from .metrics import MetricsTracker, LossTracker, EarlyStopping
from .optimizers import get_optimizer, get_scheduler, create_optimizer_with_schedule, EMA
from .losses import get_loss_function, mixup_data, cutmix_data, MixupLoss
from ..models.base_model import BaseSignLanguageModel
from ..utils.config import config


class Trainer:
    """Main trainer class for sign language detection."""
    
    def __init__(
        self,
        model: BaseSignLanguageModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        experiment_name: Optional[str] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            class_names: List of class names
            device: Device to train on
            experiment_name: Name for the experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        self.device = device or config.device
        self.model.to(self.device)
        
        # Get class names
        if class_names is None:
            self.class_names = [str(i) for i in range(model.num_classes)]
        else:
            self.class_names = class_names
        
        # Initialize trackers
        self.train_metrics = MetricsTracker(self.class_names)
        self.val_metrics = MetricsTracker(self.class_names)
        self.loss_tracker = LossTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = float('inf')  # Will be updated based on metric
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Setup experiment tracking
        self.experiment_name = experiment_name or f"sign_language_exp_{int(time.time())}"
        self.setup_logging()
        
        # Initialize training components (will be set in train method)
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.ema = None
        
        # Mixed precision training
        self.use_amp = config.get('system.mixed_precision', True)
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            self.use_amp = False
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Mixed precision training: {self.use_amp}")
    
    def setup_logging(self):
        """Setup logging and experiment tracking."""
        # Create directories
        self.log_dir = Path(config.get('logging.log_dir', 'logs')) / self.experiment_name
        self.checkpoint_dir = Path(config.get('logging.checkpoint_dir', 'checkpoints')) / self.experiment_name
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        if config.get('logging.use_tensorboard', True):
            self.tb_writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
            print(f"TensorBoard logging to: {self.log_dir / 'tensorboard'}")
        else:
            self.tb_writer = None
        
        # Setup Weights & Biases
        if config.get('logging.use_wandb', False):
            wandb.init(
                project="sign-language-detection",
                name=self.experiment_name,
                config=config._config
            )
            print("W&B logging initialized")
        
        # Save configuration
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config._config, f, indent=2)
    
    def train(
        self,
        epochs: int = 100,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        early_stopping_config: Optional[Dict[str, Any]] = None,
        use_ema: bool = False,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        save_best_only: bool = True,
        validate_every: int = 1
    ):
        """Train the model.
        
        Args:
            epochs: Number of epochs to train
            optimizer_config: Optimizer configuration
            scheduler_config: Scheduler configuration
            loss_config: Loss function configuration
            early_stopping_config: Early stopping configuration
            use_ema: Whether to use exponential moving average
            mixup_alpha: Mixup alpha parameter (0 = no mixup)
            cutmix_alpha: CutMix alpha parameter (0 = no cutmix)
            save_best_only: Whether to save only the best model
            validate_every: Validate every N epochs
        """
        print(f"Starting training for {epochs} epochs")
        
        # Setup training components
        self._setup_training_components(
            epochs=epochs,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            loss_config=loss_config,
            early_stopping_config=early_stopping_config,
            use_ema=use_ema
        )
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training phase
            train_metrics = self._train_epoch(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha
            )
            
            # Validation phase
            if epoch % validate_every == 0 or epoch == epochs - 1:
                val_metrics = self._validate_epoch()
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if hasattr(self.scheduler, 'step'):
                        if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                            self.scheduler.step(val_metrics['loss'])
                        else:
                            self.scheduler.step()
                
                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)
                
                # Check for best model
                current_val_score = val_metrics['loss']  # Can be changed to accuracy
                is_best = current_val_score < self.best_val_score
                if is_best:
                    self.best_val_score = current_val_score
                    print(f"New best validation score: {self.best_val_score:.4f}")
                
                # Save checkpoint
                if save_best_only:
                    if is_best:
                        self._save_checkpoint(epoch, is_best=True)
                else:
                    self._save_checkpoint(epoch, is_best=is_best)
                
                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(current_val_score, self.model):
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
        
        print("\nTraining completed!")
        self._save_training_summary()
        
        # Test evaluation if test loader is available
        if self.test_loader is not None:
            print("\nEvaluating on test set...")
            test_metrics = self.evaluate(self.test_loader, split_name='test')
            print(f"Test Results: {test_metrics}")
    
    def _setup_training_components(
        self,
        epochs: int,
        optimizer_config: Optional[Dict[str, Any]],
        scheduler_config: Optional[Dict[str, Any]],
        loss_config: Optional[Dict[str, Any]],
        early_stopping_config: Optional[Dict[str, Any]],
        use_ema: bool
    ):
        """Setup optimizer, scheduler, loss function, etc."""
        # Default configurations
        if optimizer_config is None:
            optimizer_config = {
                'name': config.get('training.optimizer', 'adam'),
                'lr': config.get('training.learning_rate', 0.001),
                'weight_decay': 1e-4
            }
        
        if scheduler_config is None:
            scheduler_config = {
                'name': config.get('training.scheduler', 'cosine'),
                'warmup_epochs': 5
            }
        
        if loss_config is None:
            loss_config = {'loss_type': 'cross_entropy'}
        
        # Create optimizer and scheduler
        self.optimizer, self.scheduler = create_optimizer_with_schedule(
            model=self.model,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            total_epochs=epochs,
            steps_per_epoch=len(self.train_loader)
        )
        
        # Create loss function
        self.criterion = get_loss_function(
            num_classes=self.model.num_classes,
            **loss_config
        )
        
        # Setup early stopping
        if early_stopping_config is not None:
            patience = early_stopping_config.get('patience', config.get('training.early_stopping_patience', 10))
            self.early_stopping = EarlyStopping(patience=patience)
        
        # Setup EMA
        if use_ema:
            self.ema = EMA(self.model, decay=0.999)
        
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"Loss function: {type(self.criterion).__name__}")
    
    def _train_epoch(self, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Move to device
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Apply data augmentation
            mixed_up = False
            if mixup_alpha > 0 and torch.rand(1).item() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha)
                mixed_up = True
            elif cutmix_alpha > 0 and torch.rand(1).item() < 0.5:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, cutmix_alpha)
                mixed_up = True
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if mixed_up:
                        loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                    else:
                        loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                if mixed_up:
                    loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            batch_time = time.time() - batch_start_time
            total_loss += loss.item()
            
            if not mixed_up:
                self.train_metrics.update(outputs, targets, loss.item(), batch_time)
                self.loss_tracker.update(loss.item(), self.current_epoch)
            
            # Print progress
            if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
                avg_loss = total_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Batch [{batch_idx + 1:4d}/{num_batches}] ({progress:5.1f}%) | "
                      f"Loss: {avg_loss:.4f} | Time: {batch_time:.3f}s")
        
        # Compute epoch metrics
        epoch_time = time.time() - epoch_start_time
        train_metrics = self.train_metrics.compute_metrics()
        train_metrics['epoch_time'] = epoch_time
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        return train_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        return self.evaluate(self.val_loader, split_name='val')
    
    def evaluate(self, data_loader: DataLoader, split_name: str = 'test') -> Dict[str, float]:
        """Evaluate model on given data loader."""
        self.model.eval()
        
        # Use EMA weights if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        metrics_tracker = MetricsTracker(self.class_names)
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in data_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                metrics_tracker.update(outputs, targets, loss.item())
        
        # Restore original weights if using EMA
        if self.ema is not None:
            self.ema.restore()
        
        # Compute metrics
        metrics = metrics_tracker.compute_metrics()
        
        print(f"{split_name.capitalize()} - Loss: {metrics['loss']:.4f}, "
              f"Acc: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to various tracking systems."""
        # Update training history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['val_acc'].append(val_metrics['accuracy'])
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_history['learning_rate'].append(current_lr)
        
        # TensorBoard logging
        if self.tb_writer is not None:
            # Scalars
            self.tb_writer.add_scalars('Loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, epoch)
            
            self.tb_writer.add_scalars('Accuracy', {
                'train': train_metrics['accuracy'],
                'val': val_metrics['accuracy']
            }, epoch)
            
            self.tb_writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Per-class F1 scores
            for i, (class_name, f1_score) in enumerate(zip(self.class_names, val_metrics['per_class']['f1'])):
                self.tb_writer.add_scalar(f'F1/{class_name}', f1_score, epoch)
        
        # Weights & Biases logging
        if config.get('logging.use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'learning_rate': current_lr
            })
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pth"
        if is_best:
            checkpoint_name = "best_model.pth"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        self.model.save_checkpoint(
            filepath=str(checkpoint_path),
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
            val_score=self.best_val_score,
            training_history=self.training_history
        )
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save EMA weights if available
        if self.ema is not None and is_best:
            ema_path = self.checkpoint_dir / "best_model_ema.pth"
            self.ema.apply_shadow()
            self.model.save_checkpoint(
                filepath=str(ema_path),
                epoch=epoch,
                val_score=self.best_val_score
            )
            self.ema.restore()
    
    def _save_training_summary(self):
        """Save training summary and plots."""
        summary_path = self.log_dir / 'training_summary.json'
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': self.current_epoch + 1,
            'best_val_score': self.best_val_score,
            'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0,
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else 0,
            'model_info': self.model.get_model_info()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Plot training curves
        self._plot_training_curves()
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        import matplotlib.pyplot as plt
        
        if not self.training_history['train_loss']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.training_history['train_acc'], label='Train')
        axes[0, 1].plot(epochs, self.training_history['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss difference
        loss_diff = [t - v for t, v in zip(self.training_history['train_loss'], self.training_history['val_loss'])]
        axes[1, 1].plot(epochs, loss_diff)
        axes[1, 1].set_title('Train-Val Loss Difference')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {self.log_dir / 'training_curves.png'}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_score = checkpoint.get('val_score', float('inf'))
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        # Load optimizer and scheduler states
        if load_optimizer and self.optimizer is not None:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
        print(f"Best validation score: {self.best_val_score:.4f}")
    
    def __del__(self):
        """Cleanup when trainer is destroyed."""
        if hasattr(self, 'tb_writer') and self.tb_writer is not None:
            self.tb_writer.close()
        
        if config.get('logging.use_wandb', False):
            wandb.finish()
