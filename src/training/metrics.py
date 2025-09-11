"""Metrics tracking and evaluation utilities."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, top_k_accuracy_score
)
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time


class MetricsTracker:
    """Track and compute metrics during training and evaluation."""
    
    def __init__(self, class_names: List[str]):
        """Initialize metrics tracker.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
        self.batch_times = []
        self.start_time = time.time()
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        batch_time: Optional[float] = None
    ):
        """Update metrics with batch results.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth targets
            loss: Loss value for the batch
            batch_time: Time taken for batch processing
        """
        # Convert to numpy and store
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                # Convert logits to class predictions
                predictions = torch.argmax(predictions, dim=1)
            predictions = predictions.cpu().numpy()
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        self.losses.append(loss)
        
        if batch_time is not None:
            self.batch_times.append(batch_time)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all metrics.
        
        Returns:
            Dictionary containing computed metrics
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        
        # Average loss
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        # Timing metrics
        total_time = time.time() - self.start_time
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'num_samples': len(predictions)
        }
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Get detailed classification report.
        
        Returns:
            Classification report string
        """
        if not self.predictions:
            return "No predictions available"
        
        return classification_report(
            self.targets,
            self.predictions,
            target_names=self.class_names,
            zero_division=0
        )
    
    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 10)
    ):
        """Plot confusion matrix.
        
        Args:
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.predictions:
            print("No predictions available for confusion matrix")
            return
        
        cm = confusion_matrix(
            self.targets,
            self.predictions,
            labels=range(self.num_classes)
        )
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_per_class_accuracy(self) -> Dict[str, float]:
        """Get accuracy for each class.
        
        Returns:
            Dictionary mapping class names to accuracy
        """
        if not self.predictions:
            return {}
        
        cm = confusion_matrix(
            self.targets,
            self.predictions,
            labels=range(self.num_classes)
        )
        
        per_class_acc = {}
        for i, class_name in enumerate(self.class_names):
            if cm[i].sum() > 0:
                per_class_acc[class_name] = cm[i, i] / cm[i].sum()
            else:
                per_class_acc[class_name] = 0.0
        
        return per_class_acc
    
    def get_top_k_accuracy(self, predictions_proba: torch.Tensor, k: int = 5) -> float:
        """Compute top-k accuracy.
        
        Args:
            predictions_proba: Prediction probabilities or logits
            k: Top k classes to consider
            
        Returns:
            Top-k accuracy score
        """
        if isinstance(predictions_proba, torch.Tensor):
            predictions_proba = predictions_proba.cpu().numpy()
        
        targets = np.array(self.targets)
        
        return top_k_accuracy_score(
            targets,
            predictions_proba,
            k=min(k, self.num_classes)
        )


class LossTracker:
    """Track loss values during training."""
    
    def __init__(self, window_size: int = 100):
        """Initialize loss tracker.
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.losses = []
        self.epoch_losses = defaultdict(list)
    
    def update(self, loss: float, epoch: Optional[int] = None):
        """Update loss tracker.
        
        Args:
            loss: Loss value
            epoch: Current epoch (optional)
        """
        self.losses.append(loss)
        if epoch is not None:
            self.epoch_losses[epoch].append(loss)
    
    def get_average(self) -> float:
        """Get average loss over all updates."""
        return np.mean(self.losses) if self.losses else 0.0
    
    def get_moving_average(self) -> float:
        """Get moving average loss."""
        recent_losses = self.losses[-self.window_size:]
        return np.mean(recent_losses) if recent_losses else 0.0
    
    def get_epoch_average(self, epoch: int) -> float:
        """Get average loss for specific epoch."""
        if epoch in self.epoch_losses:
            return np.mean(self.epoch_losses[epoch])
        return 0.0
    
    def plot_loss_curve(self, save_path: Optional[str] = None):
        """Plot loss curve.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.losses:
            print("No loss values to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, alpha=0.3, label='Batch Loss')
        
        # Add moving average
        if len(self.losses) > self.window_size:
            moving_avg = []
            for i in range(len(self.losses)):
                start_idx = max(0, i - self.window_size + 1)
                moving_avg.append(np.mean(self.losses[start_idx:i+1]))
            plt.plot(moving_avg, label=f'Moving Average ({self.window_size})')
        
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = 'min'
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.wait = 0
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = 0
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current score (loss or metric)
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            is_better = self.best_score is None or score < (self.best_score - self.min_delta)
        else:
            is_better = self.best_score is None or score > (self.best_score + self.min_delta)
        
        if is_better:
            self.best_score = score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {name: param.clone() for name, param in model.named_parameters()}
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights:
                # Restore best weights
                for name, param in model.named_parameters():
                    param.data.copy_(self.best_weights[name])
            return True
        
        return False


def compute_class_balanced_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute balanced accuracy (average of per-class recalls).
    
    Args:
        targets: Ground truth targets
        predictions: Model predictions
        
    Returns:
        Balanced accuracy score
    """
    cm = confusion_matrix(targets, predictions)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    # Handle classes with no samples
    per_class_acc = per_class_acc[~np.isnan(per_class_acc)]
    return np.mean(per_class_acc)


def compute_calibration_metrics(probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Compute calibration metrics for model confidence.
    
    Args:
        probabilities: Predicted probabilities
        targets: Ground truth targets
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with calibration metrics
    """
    # Get confidence (max probability) and predictions
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == targets
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0  # Expected Calibration Error
    mce = 0  # Maximum Calibration Error
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Calibration error for this bin
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            
            # ECE is weighted average
            ece += prop_in_bin * calibration_error
            
            # MCE is maximum error
            mce = max(mce, calibration_error)
    
    return {
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce
    }
