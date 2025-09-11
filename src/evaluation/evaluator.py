"""Model evaluation utilities for sign language detection."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

from ..models.base_model import BaseSignLanguageModel
from ..training.metrics import MetricsTracker, compute_calibration_metrics
from ..data.transforms import create_tta_transforms
from ..utils.config import config


class ModelEvaluator:
    """Comprehensive model evaluation toolkit."""
    
    def __init__(
        self,
        model: BaseSignLanguageModel,
        class_names: List[str],
        device: Optional[torch.device] = None
    ):
        """Initialize model evaluator.
        
        Args:
            model: Trained model to evaluate
            class_names: List of class names
            device: Device to run evaluation on
        """
        self.model = model
        self.class_names = class_names
        self.device = device or config.device
        self.model.to(self.device)
        self.model.eval()
        
        self.num_classes = len(class_names)
        self.evaluation_results = {}
    
    def evaluate_dataset(
        self,
        data_loader,
        use_tta: bool = False,
        save_predictions: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            use_tta: Whether to use test time augmentation
            save_predictions: Whether to save prediction details
            save_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating model on {len(data_loader.dataset)} samples...")
        
        # Initialize tracking
        metrics_tracker = MetricsTracker(self.class_names)
        all_predictions = []
        all_probabilities = []
        all_targets = []
        prediction_details = []
        
        # Setup TTA if requested
        if use_tta:
            tta_transforms = create_tta_transforms()
            print("Using Test Time Augmentation")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(data_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                if use_tta:
                    # Apply TTA
                    batch_preds = []
                    for img in images:
                        img_preds = []
                        # Convert tensor to numpy in correct format [H, W, C]
                        img_np = img.cpu().permute(1, 2, 0).numpy()
                        tta_imgs = tta_transforms(img_np)
                        
                        for tta_img in tta_imgs:
                            tta_img = tta_img.unsqueeze(0).to(self.device)
                            pred = torch.softmax(self.model(tta_img), dim=1)
                            img_preds.append(pred)
                        
                        # Average TTA predictions
                        avg_pred = torch.stack(img_preds).mean(dim=0)
                        batch_preds.append(avg_pred)
                    
                    outputs = torch.cat(batch_preds, dim=0)
                else:
                    # Standard inference
                    logits = self.model(images)
                    outputs = torch.softmax(logits, dim=1)
                
                # Get predictions
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Store detailed predictions if requested
                if save_predictions:
                    for i in range(len(targets)):
                        prediction_details.append({
                            'true_label': int(targets[i].cpu()),
                            'predicted_label': int(predictions[i].cpu()),
                            'true_class': self.class_names[targets[i]],
                            'predicted_class': self.class_names[predictions[i]],
                            'confidence': float(torch.max(outputs[i]).cpu()),
                            'probabilities': outputs[i].cpu().numpy().tolist()
                        })
                
                # Update metrics (using dummy loss for compatibility)
                dummy_loss = 0.0
                metrics_tracker.update(predictions, targets, dummy_loss)
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Compute comprehensive metrics
        print("Computing evaluation metrics...")
        basic_metrics = metrics_tracker.compute_metrics()
        
        # Additional metrics
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        # Calibration metrics
        calibration_metrics = compute_calibration_metrics(all_probabilities, all_targets)
        
        # Top-k accuracy
        top_5_acc = metrics_tracker.get_top_k_accuracy(
            torch.tensor(all_probabilities), k=min(5, self.num_classes)
        )
        
        # Per-class analysis
        per_class_acc = metrics_tracker.get_per_class_accuracy()
        
        # Compile results
        results = {
            **basic_metrics,
            **calibration_metrics,
            'top_5_accuracy': top_5_acc,
            'per_class_accuracy': per_class_acc,
            'use_tta': use_tta,
            'dataset_size': len(data_loader.dataset)
        }
        
        if save_predictions:
            results['prediction_details'] = prediction_details
        
        # Save results if directory provided
        if save_dir:
            self._save_evaluation_results(results, save_dir, use_tta)
        
        self.evaluation_results = results
        return results
    
    def _save_evaluation_results(self, results: Dict, save_dir: str, use_tta: bool):
        """Save evaluation results to files."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        suffix = "_tta" if use_tta else ""
        
        # Save metrics summary
        metrics_summary = {k: v for k, v in results.items() 
                          if k != 'prediction_details' and not isinstance(v, np.ndarray)}
        
        with open(save_path / f'evaluation_metrics{suffix}.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Save detailed predictions
        if 'prediction_details' in results:
            pred_df = pd.DataFrame(results['prediction_details'])
            pred_df.to_csv(save_path / f'predictions{suffix}.csv', index=False)
        
        # Save confusion matrix
        self.plot_confusion_matrix(
            save_path=save_path / f'confusion_matrix{suffix}.png'
        )
        
        # Save classification report
        report = classification_report(
            [detail['true_label'] for detail in results['prediction_details']],
            [detail['predicted_label'] for detail in results['prediction_details']],
            target_names=self.class_names,
            output_dict=True
        )
        
        with open(save_path / f'classification_report{suffix}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation results saved to {save_path}")
    
    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix from evaluation results."""
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_dataset first.")
            return
        
        # Get predictions from results
        if 'prediction_details' in self.evaluation_results:
            true_labels = [d['true_label'] for d in self.evaluation_results['prediction_details']]
            pred_labels = [d['predicted_label'] for d in self.evaluation_results['prediction_details']]
        else:
            print("No prediction details available for confusion matrix")
            return
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
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
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def analyze_misclassifications(self, top_k: int = 10) -> Dict[str, Any]:
        """Analyze the most common misclassifications.
        
        Args:
            top_k: Number of top misclassifications to analyze
            
        Returns:
            Dictionary with misclassification analysis
        """
        if 'prediction_details' in self.evaluation_results:
            details = self.evaluation_results['prediction_details']
        else:
            print("No prediction details available for misclassification analysis")
            return {}
        
        # Find misclassifications
        misclassifications = [
            d for d in details 
            if d['true_label'] != d['predicted_label']
        ]
        
        if not misclassifications:
            return {'message': 'No misclassifications found!'}
        
        # Count misclassification patterns
        error_patterns = {}
        for miscl in misclassifications:
            pattern = (miscl['true_class'], miscl['predicted_class'])
            if pattern not in error_patterns:
                error_patterns[pattern] = {
                    'count': 0,
                    'confidences': [],
                    'examples': []
                }
            error_patterns[pattern]['count'] += 1
            error_patterns[pattern]['confidences'].append(miscl['confidence'])
            if len(error_patterns[pattern]['examples']) < 5:
                error_patterns[pattern]['examples'].append(miscl)
        
        # Sort by frequency
        sorted_patterns = sorted(
            error_patterns.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        # Analyze top patterns
        analysis = {
            'total_misclassifications': len(misclassifications),
            'total_samples': len(details),
            'error_rate': len(misclassifications) / len(details),
            'top_error_patterns': []
        }
        
        for (true_class, pred_class), info in sorted_patterns[:top_k]:
            pattern_analysis = {
                'true_class': true_class,
                'predicted_class': pred_class,
                'count': info['count'],
                'percentage_of_errors': info['count'] / len(misclassifications) * 100,
                'avg_confidence': np.mean(info['confidences']),
                'std_confidence': np.std(info['confidences'])
            }
            analysis['top_error_patterns'].append(pattern_analysis)
        
        return analysis
    
    def analyze_confidence_distribution(self) -> Dict[str, Any]:
        """Analyze confidence distribution of predictions."""
        if 'prediction_details' not in self.evaluation_results:
            print("No prediction details available for confidence analysis")
            return {}
        
        details = self.evaluation_results['prediction_details']
        
        # Separate correct and incorrect predictions
        correct_confidences = [
            d['confidence'] for d in details 
            if d['true_label'] == d['predicted_label']
        ]
        
        incorrect_confidences = [
            d['confidence'] for d in details 
            if d['true_label'] != d['predicted_label']
        ]
        
        analysis = {
            'correct_predictions': {
                'count': len(correct_confidences),
                'mean_confidence': np.mean(correct_confidences) if correct_confidences else 0,
                'std_confidence': np.std(correct_confidences) if correct_confidences else 0,
                'confidence_ranges': self._compute_confidence_ranges(correct_confidences)
            },
            'incorrect_predictions': {
                'count': len(incorrect_confidences),
                'mean_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0,
                'std_confidence': np.std(incorrect_confidences) if incorrect_confidences else 0,
                'confidence_ranges': self._compute_confidence_ranges(incorrect_confidences)
            }
        }
        
        return analysis
    
    def _compute_confidence_ranges(self, confidences: List[float]) -> Dict[str, int]:
        """Compute confidence range distribution."""
        if not confidences:
            return {}
        
        ranges = {
            '0.0-0.5': 0,
            '0.5-0.7': 0,
            '0.7-0.8': 0,
            '0.8-0.9': 0,
            '0.9-1.0': 0
        }
        
        for conf in confidences:
            if conf < 0.5:
                ranges['0.0-0.5'] += 1
            elif conf < 0.7:
                ranges['0.5-0.7'] += 1
            elif conf < 0.8:
                ranges['0.7-0.8'] += 1
            elif conf < 0.9:
                ranges['0.8-0.9'] += 1
            else:
                ranges['0.9-1.0'] += 1
        
        return ranges
    
    def plot_confidence_distribution(self, save_path: Optional[str] = None):
        """Plot confidence distribution for correct and incorrect predictions."""
        if 'prediction_details' not in self.evaluation_results:
            print("No prediction details available")
            return
        
        details = self.evaluation_results['prediction_details']
        
        correct_confidences = [
            d['confidence'] for d in details 
            if d['true_label'] == d['predicted_label']
        ]
        
        incorrect_confidences = [
            d['confidence'] for d in details 
            if d['true_label'] != d['predicted_label']
        ]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=20, alpha=0.7, color='green', label='Correct')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, color='red', label='Incorrect')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        ranges = ['0.0-0.5', '0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        correct_ranges = self._compute_confidence_ranges(correct_confidences)
        incorrect_ranges = self._compute_confidence_ranges(incorrect_confidences)
        
        x = np.arange(len(ranges))
        width = 0.35
        
        plt.bar(x - width/2, [correct_ranges.get(r, 0) for r in ranges], 
                width, label='Correct', color='green', alpha=0.7)
        plt.bar(x + width/2, [incorrect_ranges.get(r, 0) for r in ranges], 
                width, label='Incorrect', color='red', alpha=0.7)
        
        plt.xlabel('Confidence Range')
        plt.ylabel('Count')
        plt.title('Confidence Range Distribution')
        plt.xticks(x, ranges, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence distribution plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, save_dir: str) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            save_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_dataset first.")
            return ""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        report_path = save_path / 'evaluation_report.md'
        
        # Generate report content
        report_content = self._generate_report_content()
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Evaluation report saved to {report_path}")
        return str(report_path)
    
    def _generate_report_content(self) -> str:
        """Generate markdown report content."""
        results = self.evaluation_results
        
        # Get misclassification and confidence analysis
        miscl_analysis = self.analyze_misclassifications()
        conf_analysis = self.analyze_confidence_distribution()
        
        report = f"""# Model Evaluation Report

## Overview
- **Model Architecture**: {self.model.__class__.__name__}
- **Number of Classes**: {self.num_classes}
- **Dataset Size**: {results.get('dataset_size', 'N/A')}
- **Test Time Augmentation**: {results.get('use_tta', False)}

## Performance Metrics

### Overall Performance
- **Accuracy**: {results.get('accuracy', 0):.4f}
- **Top-5 Accuracy**: {results.get('top_5_accuracy', 0):.4f}
- **F1-Score (Macro)**: {results.get('f1_macro', 0):.4f}
- **F1-Score (Weighted)**: {results.get('f1_weighted', 0):.4f}

### Precision and Recall
- **Precision (Macro)**: {results.get('precision_macro', 0):.4f}
- **Recall (Macro)**: {results.get('recall_macro', 0):.4f}
- **Precision (Weighted)**: {results.get('precision_weighted', 0):.4f}
- **Recall (Weighted)**: {results.get('recall_weighted', 0):.4f}

### Calibration Metrics
- **Expected Calibration Error**: {results.get('expected_calibration_error', 0):.4f}
- **Maximum Calibration Error**: {results.get('maximum_calibration_error', 0):.4f}

## Per-Class Performance
"""
        
        # Add per-class metrics
        per_class_acc = results.get('per_class_accuracy', {})
        for class_name, acc in sorted(per_class_acc.items()):
            report += f"- **{class_name}**: {acc:.4f}\n"
        
        # Add misclassification analysis
        if miscl_analysis and 'top_error_patterns' in miscl_analysis:
            report += f"""
## Error Analysis

### Overall Error Statistics
- **Total Misclassifications**: {miscl_analysis.get('total_misclassifications', 0)}
- **Error Rate**: {miscl_analysis.get('error_rate', 0):.4f}

### Top Misclassification Patterns
"""
            for pattern in miscl_analysis['top_error_patterns'][:5]:
                report += f"- **{pattern['true_class']} → {pattern['predicted_class']}**: {pattern['count']} errors ({pattern['percentage_of_errors']:.1f}% of all errors)\n"
        
        # Add confidence analysis
        if conf_analysis:
            report += f"""
## Confidence Analysis

### Correct Predictions
- **Count**: {conf_analysis['correct_predictions']['count']}
- **Mean Confidence**: {conf_analysis['correct_predictions']['mean_confidence']:.4f}

### Incorrect Predictions
- **Count**: {conf_analysis['incorrect_predictions']['count']}
- **Mean Confidence**: {conf_analysis['incorrect_predictions']['mean_confidence']:.4f}

## Recommendations

Based on the evaluation results:
"""
            
            # Add recommendations based on metrics
            acc = results.get('accuracy', 0)
            if acc > 0.95:
                report += "- Excellent model performance! Consider deploying to production.\n"
            elif acc > 0.90:
                report += "- Good model performance. Consider fine-tuning for better results.\n"
            elif acc > 0.80:
                report += "- Moderate performance. Consider data augmentation or model architecture changes.\n"
            else:
                report += "- Poor performance. Revisit data quality, model architecture, and training strategy.\n"
            
            # Calibration recommendations
            ece = results.get('expected_calibration_error', 0)
            if ece > 0.1:
                report += "- Model is poorly calibrated. Consider calibration techniques like temperature scaling.\n"
            
            # Class imbalance recommendations
            if len(set(per_class_acc.values())) > 1:  # Different accuracies across classes
                worst_classes = sorted(per_class_acc.items(), key=lambda x: x[1])[:3]
                report += f"- Focus on improving performance for: {', '.join([c[0] for c in worst_classes])}\n"
        
        report += f"""
---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
