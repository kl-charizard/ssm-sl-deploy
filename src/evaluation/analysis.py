"""Advanced model analysis and comparison utilities."""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2

from ..models.base_model import BaseSignLanguageModel
from .evaluator import ModelEvaluator


class ModelAnalysis:
    """Advanced analysis toolkit for trained models."""
    
    def __init__(
        self,
        model: BaseSignLanguageModel,
        class_names: List[str],
        device: Optional[torch.device] = None
    ):
        """Initialize model analysis.
        
        Args:
            model: Trained model to analyze
            class_names: List of class names
            device: Device to run analysis on
        """
        self.model = model
        self.class_names = class_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_layer_activations(
        self,
        data_loader,
        layer_names: Optional[List[str]] = None,
        max_samples: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Analyze activations in different layers.
        
        Args:
            data_loader: DataLoader for input data
            layer_names: Names of layers to analyze
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = []
                # Take mean across spatial dimensions if conv layer
                if len(output.shape) == 4:  # Conv output
                    act = output.mean(dim=[2, 3]).detach().cpu().numpy()
                else:  # FC output
                    act = output.detach().cpu().numpy()
                activations[name].append(act)
            return hook
        
        # Register hooks
        hooks = []
        if layer_names is None:
            # Default to some common layer types
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
        else:
            model_dict = dict(self.model.named_modules())
            for layer_name in layer_names:
                if layer_name in model_dict:
                    hook = model_dict[layer_name].register_forward_hook(hook_fn(layer_name))
                    hooks.append(hook)
        
        # Forward pass to collect activations
        sample_count = 0
        with torch.no_grad():
            for images, _ in data_loader:
                if sample_count >= max_samples:
                    break
                
                images = images.to(self.device)
                _ = self.model(images)
                sample_count += len(images)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate activations
        for name in activations:
            activations[name] = np.concatenate(activations[name], axis=0)
        
        return activations
    
    def visualize_feature_space(
        self,
        data_loader,
        layer_name: Optional[str] = None,
        method: str = 'tsne',
        max_samples: int = 1000,
        save_path: Optional[str] = None
    ):
        """Visualize feature space using dimensionality reduction.
        
        Args:
            data_loader: DataLoader for input data
            layer_name: Name of layer to extract features from
            method: Dimensionality reduction method ('tsne', 'pca')
            max_samples: Maximum number of samples
            save_path: Path to save the plot
        """
        # Extract features
        features = []
        labels = []
        
        with torch.no_grad():
            sample_count = 0
            for images, targets in data_loader:
                if sample_count >= max_samples:
                    break
                
                images = images.to(self.device)
                
                # Get features from specified layer or final layer
                if layer_name and hasattr(self.model, 'get_feature_maps'):
                    feat = self.model.get_feature_maps(images, layer_name)
                else:
                    # Use features before final classifier
                    feat = self._extract_features(images)
                
                if feat is not None:
                    if len(feat.shape) > 2:
                        feat = feat.mean(dim=list(range(2, len(feat.shape))))  # Global average pooling
                    
                    features.append(feat.cpu().numpy())
                    labels.extend(targets.cpu().numpy())
                    sample_count += len(images)
        
        if not features:
            print("Could not extract features")
            return
        
        features = np.concatenate(features, axis=0)
        labels = np.array(labels)
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        features_2d = reducer.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=labels, cmap='tab20', alpha=0.7
        )
        plt.colorbar(scatter, label='Class')
        plt.title(f'Feature Space Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # Add class names to legend if not too many classes
        if len(self.class_names) <= 20:
            handles, _ = scatter.legend_elements()
            plt.legend(handles, self.class_names, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature space plot saved to {save_path}")
        
        plt.show()
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from the model."""
        # This is a generic feature extraction method
        # For specific models, you might need to customize this
        
        if hasattr(self.model, 'features'):
            # For models with feature extractor
            features = self.model.features(images)
            return torch.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        elif hasattr(self.model, 'backbone'):
            # For pretrained models
            features = self.model.backbone.features(images)
            return torch.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        else:
            # Generic approach - use intermediate layer
            for name, module in self.model.named_children():
                if isinstance(module, torch.nn.Linear):
                    # Stop before final classifier
                    break
                images = module(images)
            
            if len(images.shape) > 2:
                images = torch.adaptive_avg_pool2d(images, (1, 1)).flatten(1)
            
            return images
    
    def analyze_gradients(
        self,
        data_loader,
        target_class: Optional[int] = None,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Analyze gradients for interpretability.
        
        Args:
            data_loader: DataLoader for input data
            target_class: Target class for gradient analysis
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dictionary with gradient analysis results
        """
        self.model.eval()
        gradient_norms = []
        
        sample_count = 0
        for images, targets in data_loader:
            if sample_count >= max_samples:
                break
            
            images = images.to(self.device)
            targets = targets.to(self.device)
            images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(images)
            
            # Use target class or predicted class
            if target_class is not None:
                target_scores = outputs[:, target_class]
            else:
                target_scores = outputs[range(len(outputs)), targets]
            
            # Backward pass
            grad_outputs = torch.ones_like(target_scores)
            gradients = torch.autograd.grad(
                outputs=target_scores,
                inputs=images,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Calculate gradient norms
            grad_norms = torch.norm(gradients.view(len(gradients), -1), dim=1)
            gradient_norms.extend(grad_norms.cpu().numpy())
            
            sample_count += len(images)
        
        gradient_norms = np.array(gradient_norms)
        
        analysis = {
            'mean_gradient_norm': float(np.mean(gradient_norms)),
            'std_gradient_norm': float(np.std(gradient_norms)),
            'median_gradient_norm': float(np.median(gradient_norms)),
            'max_gradient_norm': float(np.max(gradient_norms)),
            'min_gradient_norm': float(np.min(gradient_norms))
        }
        
        return analysis
    
    def generate_grad_cam(
        self,
        image: torch.Tensor,
        target_class: int,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for an image.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index
            layer_name: Name of layer to use for Grad-CAM
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Find target layer
        if layer_name is None:
            # Use the last convolutional layer
            target_layer = None
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
            if target_layer is None:
                print("No convolutional layer found for Grad-CAM")
                return np.zeros((224, 224))
        else:
            target_layer = dict(self.model.named_modules())[layer_name]
        
        # Hook to capture gradients and activations
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        # Register hooks
        backward_handle = target_layer.register_backward_hook(backward_hook)
        forward_handle = target_layer.register_forward_hook(forward_hook)
        
        try:
            # Forward pass
            image = image.to(self.device)
            image.requires_grad_(True)
            outputs = self.model(image)
            
            # Backward pass for target class
            outputs[0, target_class].backward()
            
            # Get gradients and activations
            grads = gradients[0][0]  # First sample
            acts = activations[0][0]  # First sample
            
            # Compute weights (global average pooling of gradients)
            weights = torch.mean(grads, dim=[1, 2])
            
            # Compute Grad-CAM
            grad_cam = torch.zeros(acts.shape[1:], device=self.device)
            for i, w in enumerate(weights):
                grad_cam += w * acts[i]
            
            # Apply ReLU
            grad_cam = torch.relu(grad_cam)
            
            # Normalize
            grad_cam = grad_cam / torch.max(grad_cam)
            
            # Resize to input image size
            grad_cam = torch.nn.functional.interpolate(
                grad_cam.unsqueeze(0).unsqueeze(0),
                size=(image.shape[2], image.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            
            return grad_cam.squeeze().cpu().numpy()
        
        finally:
            # Remove hooks
            backward_handle.remove()
            forward_handle.remove()
    
    def compare_predictions_on_similar_images(
        self,
        data_loader,
        similarity_threshold: float = 0.9,
        max_pairs: int = 50
    ) -> List[Dict[str, Any]]:
        """Compare predictions on similar images to find inconsistencies.
        
        Args:
            data_loader: DataLoader for input data
            similarity_threshold: Minimum similarity for image pairs
            max_pairs: Maximum number of pairs to analyze
            
        Returns:
            List of inconsistent prediction pairs
        """
        # Extract features and predictions
        features = []
        predictions = []
        labels = []
        images_data = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                if len(features) >= max_pairs * 10:  # Collect more data for finding pairs
                    break
                
                images = images.to(self.device)
                feat = self._extract_features(images)
                pred = self.model(images)
                
                features.append(feat.cpu())
                predictions.append(pred.cpu())
                labels.extend(targets.numpy())
                images_data.append(images.cpu())
        
        if not features:
            return []
        
        features = torch.cat(features, dim=0)
        predictions = torch.cat(predictions, dim=0)
        images_data = torch.cat(images_data, dim=0)
        labels = np.array(labels)
        
        # Normalize features for cosine similarity
        features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # Find similar pairs
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        inconsistent_pairs = []
        found_pairs = 0
        
        for i in range(len(similarity_matrix)):
            if found_pairs >= max_pairs:
                break
            
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i, j] > similarity_threshold:
                    # Check if predictions are different
                    pred_i = torch.argmax(predictions[i])
                    pred_j = torch.argmax(predictions[j])
                    
                    if pred_i != pred_j or (pred_i != labels[i] and pred_j != labels[j]):
                        inconsistent_pairs.append({
                            'image_1_idx': i,
                            'image_2_idx': j,
                            'similarity': float(similarity_matrix[i, j]),
                            'label_1': int(labels[i]),
                            'label_2': int(labels[j]),
                            'pred_1': int(pred_i),
                            'pred_2': int(pred_j),
                            'confidence_1': float(torch.max(torch.softmax(predictions[i], dim=0))),
                            'confidence_2': float(torch.max(torch.softmax(predictions[j], dim=0)))
                        })
                        found_pairs += 1
                        
                        if found_pairs >= max_pairs:
                            break
        
        return inconsistent_pairs


def compare_models(
    models: List[BaseSignLanguageModel],
    model_names: List[str],
    test_loader,
    class_names: List[str],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Compare multiple models on the same test set.
    
    Args:
        models: List of models to compare
        model_names: Names for each model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run comparison on
        
    Returns:
        Dictionary with comparison results
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    all_predictions = {}
    all_probabilities = {}
    
    # Evaluate each model
    for model, name in zip(models, model_names):
        print(f"Evaluating {name}...")
        evaluator = ModelEvaluator(model, class_names, device)
        result = evaluator.evaluate_dataset(test_loader, save_predictions=True, save_dir=None)
        results[name] = result
        
        # Store predictions for agreement analysis
        if 'prediction_details' in result:
            all_predictions[name] = [d['predicted_label'] for d in result['prediction_details']]
            all_probabilities[name] = [d['probabilities'] for d in result['prediction_details']]
    
    # Compute agreement metrics
    agreement_analysis = _compute_model_agreement(all_predictions, model_names)
    
    # Find ensemble potential
    ensemble_analysis = _analyze_ensemble_potential(all_probabilities, model_names, test_loader)
    
    # Create comparison summary
    comparison = {
        'individual_results': results,
        'agreement_analysis': agreement_analysis,
        'ensemble_analysis': ensemble_analysis,
        'summary': _create_comparison_summary(results, model_names)
    }
    
    return comparison


def _compute_model_agreement(predictions: Dict[str, List[int]], model_names: List[str]) -> Dict[str, Any]:
    """Compute agreement metrics between models."""
    agreement_matrix = np.zeros((len(model_names), len(model_names)))
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                pred1 = np.array(predictions[name1])
                pred2 = np.array(predictions[name2])
                agreement = np.mean(pred1 == pred2)
                agreement_matrix[i, j] = agreement
    
    # Find most agreeable pairs
    pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            pairs.append({
                'model_1': model_names[i],
                'model_2': model_names[j],
                'agreement': agreement_matrix[i, j]
            })
    
    pairs.sort(key=lambda x: x['agreement'], reverse=True)
    
    return {
        'agreement_matrix': agreement_matrix.tolist(),
        'model_names': model_names,
        'top_agreeable_pairs': pairs[:5]
    }


def _analyze_ensemble_potential(
    probabilities: Dict[str, List[List[float]]],
    model_names: List[str],
    test_loader
) -> Dict[str, Any]:
    """Analyze potential benefits of ensembling models."""
    # Simple ensemble (average probabilities)
    ensemble_probs = []
    
    if probabilities:
        num_samples = len(list(probabilities.values())[0])
        num_classes = len(probabilities[model_names[0]][0])
        
        for i in range(num_samples):
            sample_probs = []
            for model_name in model_names:
                sample_probs.append(probabilities[model_name][i])
            
            # Average probabilities
            avg_probs = np.mean(sample_probs, axis=0)
            ensemble_probs.append(avg_probs)
        
        # Get true labels
        true_labels = []
        for _, targets in test_loader:
            true_labels.extend(targets.numpy())
        
        # Compute ensemble accuracy
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        ensemble_accuracy = np.mean(ensemble_predictions == np.array(true_labels[:len(ensemble_predictions)]))
        
        # Compare with individual model accuracies
        individual_accuracies = {}
        for model_name in model_names:
            preds = np.argmax(probabilities[model_name], axis=1)
            acc = np.mean(preds == np.array(true_labels[:len(preds)]))
            individual_accuracies[model_name] = acc
        
        best_individual = max(individual_accuracies.values())
        improvement = ensemble_accuracy - best_individual
        
        return {
            'ensemble_accuracy': float(ensemble_accuracy),
            'individual_accuracies': individual_accuracies,
            'best_individual_accuracy': float(best_individual),
            'ensemble_improvement': float(improvement)
        }
    
    return {}


def _create_comparison_summary(results: Dict[str, Dict], model_names: List[str]) -> Dict[str, Any]:
    """Create a summary comparison of models."""
    summary = {
        'best_accuracy': {'model': '', 'value': 0.0},
        'best_f1_macro': {'model': '', 'value': 0.0},
        'fastest_inference': {'model': '', 'value': float('inf')},
        'model_rankings': {}
    }
    
    # Find best performers
    for name in model_names:
        if name in results:
            result = results[name]
            
            # Best accuracy
            acc = result.get('accuracy', 0)
            if acc > summary['best_accuracy']['value']:
                summary['best_accuracy'] = {'model': name, 'value': acc}
            
            # Best F1
            f1 = result.get('f1_macro', 0)
            if f1 > summary['best_f1_macro']['value']:
                summary['best_f1_macro'] = {'model': name, 'value': f1}
            
            # Fastest inference (if timing info available)
            avg_time = result.get('avg_batch_time', float('inf'))
            if avg_time < summary['fastest_inference']['value']:
                summary['fastest_inference'] = {'model': name, 'value': avg_time}
    
    # Create rankings
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    for metric in metrics:
        ranking = []
        for name in model_names:
            if name in results:
                value = results[name].get(metric, 0)
                ranking.append({'model': name, 'value': value})
        
        ranking.sort(key=lambda x: x['value'], reverse=True)
        summary['model_rankings'][metric] = ranking
    
    return summary
