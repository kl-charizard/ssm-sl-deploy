"""Inference engine for sign language detection."""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import time
from collections import deque
import threading

from ..models.base_model import BaseSignLanguageModel
from ..data.transforms import get_transforms
from ..utils.config import config


class InferenceEngine:
    """High-performance inference engine for real-time sign language detection."""
    
    def __init__(
        self,
        model: BaseSignLanguageModel,
        class_names: List[str],
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.5,
        smoothing_window: int = 5
    ):
        """Initialize inference engine.
        
        Args:
            model: Trained model for inference
            class_names: List of class names
            device: Device to run inference on
            confidence_threshold: Minimum confidence for predictions
            smoothing_window: Window size for prediction smoothing
        """
        self.model = model
        self.class_names = class_names
        self.device = device or config.device
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        input_size = config.get('model.input_size', [224, 224])
        self.transform = get_transforms(input_size=input_size, mode='test')
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.frame_count = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        print(f"Inference engine initialized on {self.device}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Classes: {len(class_names)}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Convert BGR to RGB if needed
        if image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if hasattr(self.transform, '__call__'):
            # Handle both PIL and tensor transforms
            try:
                from PIL import Image
                pil_image = Image.fromarray(image)
                tensor = self.transform(pil_image)
            except:
                # Fallback to direct tensor conversion
                tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                # Apply normalization manually
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = (tensor - mean) / std
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict_single(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform single image prediction.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess
        tensor = self.preprocess_image(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        inference_time = time.time() - start_time
        
        # Convert to numpy
        predicted_class = int(predicted.item())
        confidence_score = float(confidence.item())
        prob_array = probabilities.cpu().numpy()[0]
        
        # Create result
        result = {
            'predicted_class_idx': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': confidence_score,
            'probabilities': prob_array,
            'inference_time': inference_time,
            'above_threshold': confidence_score >= self.confidence_threshold
        }
        
        # Track performance
        with self.lock:
            self.inference_times.append(inference_time)
            self.frame_count += 1
        
        return result
    
    def predict_with_smoothing(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform prediction with temporal smoothing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with smoothed prediction results
        """
        # Get current prediction
        current_result = self.predict_single(image)
        
        # Add to history
        self.prediction_history.append(current_result['probabilities'])
        self.confidence_history.append(current_result['confidence'])
        
        # Compute smoothed prediction
        if len(self.prediction_history) >= 2:
            # Average probabilities over history
            avg_probabilities = np.mean(list(self.prediction_history), axis=0)
            smoothed_predicted = np.argmax(avg_probabilities)
            smoothed_confidence = float(avg_probabilities[smoothed_predicted])
            
            # Smooth confidence as well
            avg_confidence = np.mean(list(self.confidence_history))
            
            smoothed_result = {
                'predicted_class_idx': int(smoothed_predicted),
                'predicted_class_name': self.class_names[smoothed_predicted],
                'confidence': smoothed_confidence,
                'avg_confidence': avg_confidence,
                'probabilities': avg_probabilities,
                'inference_time': current_result['inference_time'],
                'above_threshold': smoothed_confidence >= self.confidence_threshold,
                'raw_prediction': current_result
            }
        else:
            smoothed_result = current_result.copy()
            smoothed_result['avg_confidence'] = current_result['confidence']
            smoothed_result['raw_prediction'] = current_result
        
        return smoothed_result
    
    def get_top_k_predictions(
        self,
        image: np.ndarray,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top-k predictions for an image.
        
        Args:
            image: Input image as numpy array
            k: Number of top predictions to return
            
        Returns:
            List of top-k predictions
        """
        result = self.predict_single(image)
        probabilities = result['probabilities']
        
        # Get top-k indices
        top_k_indices = np.argsort(probabilities)[::-1][:k]
        
        top_predictions = []
        for idx in top_k_indices:
            top_predictions.append({
                'class_idx': int(idx),
                'class_name': self.class_names[idx],
                'probability': float(probabilities[idx])
            })
        
        return top_predictions
    
    def reset_smoothing(self):
        """Reset prediction smoothing history."""
        self.prediction_history.clear()
        self.confidence_history.clear()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            if not self.inference_times:
                return {}
            
            times = list(self.inference_times)
            
            stats = {
                'avg_inference_time': float(np.mean(times)),
                'min_inference_time': float(np.min(times)),
                'max_inference_time': float(np.max(times)),
                'std_inference_time': float(np.std(times)),
                'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
                'total_frames': self.frame_count
            }
        
        return stats
    
    def benchmark(self, num_iterations: int = 100, image_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
        """Benchmark inference performance.
        
        Args:
            num_iterations: Number of inference iterations
            image_size: Size of test images
            
        Returns:
            Benchmark results
        """
        print(f"Benchmarking inference engine for {num_iterations} iterations...")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            self.predict_single(dummy_image)
        
        # Benchmark
        times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            self.predict_single(dummy_image)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations")
        
        # Calculate statistics
        times = np.array(times)
        
        results = {
            'num_iterations': num_iterations,
            'total_time': float(np.sum(times)),
            'avg_time_per_inference': float(np.mean(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'std_time': float(np.std(times)),
            'median_time': float(np.median(times)),
            'p95_time': float(np.percentile(times, 95)),
            'p99_time': float(np.percentile(times, 99)),
            'avg_fps': float(1.0 / np.mean(times)),
            'device': str(self.device),
            'model_name': self.model.__class__.__name__
        }
        
        print(f"Benchmark completed!")
        print(f"Average inference time: {results['avg_time_per_inference']*1000:.2f} ms")
        print(f"Average FPS: {results['avg_fps']:.1f}")
        
        return results
    
    def process_video_stream(
        self,
        video_source: Any,
        callback_fn: Optional[callable] = None,
        display: bool = True,
        max_fps: float = 30.0
    ):
        """Process video stream with real-time inference.
        
        Args:
            video_source: Video source (camera index, file path, or cv2.VideoCapture)
            callback_fn: Callback function to handle predictions
            display: Whether to display the video with predictions
            max_fps: Maximum FPS for processing
        """
        # Open video source
        if isinstance(video_source, (int, str)):
            cap = cv2.VideoCapture(video_source)
        else:
            cap = video_source
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        frame_interval = 1.0 / max_fps
        last_process_time = 0
        
        print("Starting video stream processing. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Control FPS
                if current_time - last_process_time >= frame_interval:
                    # Process frame
                    result = self.predict_with_smoothing(frame)
                    
                    # Call callback if provided
                    if callback_fn:
                        callback_fn(frame, result)
                    
                    # Display if requested
                    if display:
                        self._draw_prediction_on_frame(frame, result)
                        cv2.imshow('Sign Language Detection', frame)
                    
                    last_process_time = current_time
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _draw_prediction_on_frame(self, frame: np.ndarray, result: Dict[str, Any]):
        """Draw prediction results on frame.
        
        Args:
            frame: Input frame to draw on
            result: Prediction results
        """
        height, width = frame.shape[:2]
        
        # Determine color based on confidence
        if result['above_threshold']:
            color = (0, 255, 0)  # Green for confident predictions
        else:
            color = (0, 165, 255)  # Orange for uncertain predictions
        
        # Draw prediction text
        text = f"{result['predicted_class_name']}: {result['confidence']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (text_width + 20, text_height + 20), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (15, text_height + 15), font, font_scale, color, thickness)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = width - bar_width - 10
        bar_y = 10
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Confidence bar
        conf_width = int(bar_width * result['confidence'])
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), color, -1)
        
        # Threshold line
        thresh_x = int(bar_x + bar_width * self.confidence_threshold)
        cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_height), (255, 255, 255), 2)
        
        # FPS counter
        stats = self.get_performance_stats()
        if stats:
            fps_text = f"FPS: {stats['fps']:.1f}"
            cv2.putText(frame, fps_text, (width - 100, height - 20), font, 0.5, (255, 255, 255), 1)


class BatchInferenceEngine:
    """Batch inference engine for processing multiple images efficiently."""
    
    def __init__(
        self,
        model: BaseSignLanguageModel,
        class_names: List[str],
        device: Optional[torch.device] = None,
        batch_size: int = 32
    ):
        """Initialize batch inference engine.
        
        Args:
            model: Trained model for inference
            class_names: List of class names
            device: Device to run inference on
            batch_size: Batch size for processing
        """
        self.model = model
        self.class_names = class_names
        self.device = device or config.device
        self.batch_size = batch_size
        
        self.model.to(self.device)
        self.model.eval()
        
        input_size = config.get('model.input_size', [224, 224])
        self.transform = get_transforms(input_size=input_size, mode='test')
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Predict on a batch of images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_results = self._process_batch(batch_images)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process a single batch of images."""
        # Preprocess batch
        batch_tensors = []
        for image in images:
            tensor = self._preprocess_image(image)
            batch_tensors.append(tensor)
        
        if not batch_tensors:
            return []
        
        # Stack into batch
        batch = torch.cat(batch_tensors, dim=0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(batch)
            probabilities = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
        
        # Convert results
        results = []
        for i in range(len(images)):
            result = {
                'predicted_class_idx': int(predictions[i].item()),
                'predicted_class_name': self.class_names[predictions[i].item()],
                'confidence': float(confidences[i].item()),
                'probabilities': probabilities[i].cpu().numpy()
            }
            results.append(result)
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess single image."""
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            from PIL import Image
            pil_image = Image.fromarray(image)
            tensor = self.transform(pil_image)
        except:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return tensor.unsqueeze(0)
