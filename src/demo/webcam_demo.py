"""Real-time webcam demo for sign language detection."""

import cv2
import numpy as np
import argparse
import time
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import threading
import queue

from .inference_engine import InferenceEngine
from ..models.model_factory import create_model, load_model_from_config
from ..utils.config import config, Config


class WebcamDemo:
    """Real-time webcam demo application."""
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        camera_index: int = 0,
        confidence_threshold: float = 0.5,
        smoothing_window: int = 5,
        display_size: tuple = (800, 600)
    ):
        """Initialize webcam demo.
        
        Args:
            model_path: Path to trained model checkpoint
            class_names: List of class names
            camera_index: Camera device index
            confidence_threshold: Minimum confidence for predictions
            smoothing_window: Window size for prediction smoothing
            display_size: Display window size (width, height)
        """
        self.model_path = model_path
        self.class_names = class_names
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.display_size = display_size
        
        # Initialize model and inference engine
        self.model = None
        self.inference_engine = None
        self.load_model()
        
        # Camera setup
        self.cap = None
        self.setup_camera()
        
        # Demo state
        self.is_running = False
        self.show_debug_info = True
        self.show_top_k = 3
        self.recording = False
        self.prediction_history = []
        
        # Threading for smooth playback
        self.frame_queue = queue.Queue(maxsize=2)
        self.prediction_queue = queue.Queue(maxsize=2)
        
        print(f"Webcam demo initialized")
        print(f"Camera: {camera_index}, Model: {Path(model_path).name}")
        print(f"Classes: {len(class_names)}")
    
    def load_model(self):
        """Load the trained model."""
        try:
            print(f"Loading model from {self.model_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Get model configuration
            model_info = checkpoint.get('model_info', {})
            num_classes = len(self.class_names)
            
            # Create model (you might need to adjust this based on your checkpoint structure)
            # This assumes the checkpoint contains enough info to recreate the model
            if 'model_config' in checkpoint:
                self.model = load_model_from_config(
                    checkpoint['model_config'],
                    num_classes=num_classes,
                    checkpoint_path=self.model_path
                )
            else:
                # Fallback: try to infer model type
                architecture = config.get('model.architecture', 'efficientnet_b0')
                self.model = create_model(
                    architecture=architecture,
                    num_classes=num_classes,
                    pretrained=False
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Initialize inference engine
            self.inference_engine = InferenceEngine(
                model=self.model,
                class_names=self.class_names,
                confidence_threshold=self.confidence_threshold,
                smoothing_window=self.smoothing_window
            )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def setup_camera(self):
        """Setup camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"Camera {self.camera_index} initialized")
            
        except Exception as e:
            print(f"Error setting up camera: {e}")
            raise
    
    def capture_thread(self):
        """Thread function for capturing frames."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Put frame in queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            else:
                time.sleep(0.01)
    
    def inference_thread(self):
        """Thread function for running inference."""
        while self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Run inference
                result = self.inference_engine.predict_with_smoothing(frame)
                
                # Store prediction history
                if self.recording:
                    self.prediction_history.append({
                        'timestamp': time.time(),
                        'prediction': result['predicted_class_name'],
                        'confidence': result['confidence']
                    })
                
                # Put result in queue (non-blocking)
                if not self.prediction_queue.full():
                    self.prediction_queue.put((frame, result))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")
                continue
    
    def draw_ui(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw user interface on frame.
        
        Args:
            frame: Input frame
            result: Prediction result
            
        Returns:
            Frame with UI elements drawn
        """
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Main prediction
        self._draw_main_prediction(overlay, result)
        
        # Top-k predictions
        if self.show_debug_info:
            top_k_preds = self.inference_engine.get_top_k_predictions(frame, k=self.show_top_k)
            self._draw_top_k_predictions(overlay, top_k_preds)
        
        # Performance info
        if self.show_debug_info:
            self._draw_performance_info(overlay)
        
        # Controls info
        self._draw_controls_info(overlay)
        
        # Recording indicator
        if self.recording:
            self._draw_recording_indicator(overlay)
        
        return overlay
    
    def _draw_main_prediction(self, frame: np.ndarray, result: Dict[str, Any]):
        """Draw main prediction on frame."""
        height, width = frame.shape[:2]
        
        # Determine color based on confidence
        if result['above_threshold']:
            color = (0, 255, 0)  # Green
            bg_color = (0, 100, 0)
        else:
            color = (0, 165, 255)  # Orange
            bg_color = (0, 65, 100)
        
        # Main prediction text
        text = f"{result['predicted_class_name']}"
        conf_text = f"{result['confidence']:.2f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        (conf_width, conf_height), _ = cv2.getTextSize(conf_text, font, 1.0, 2)
        
        # Draw background
        padding = 20
        bg_width = max(text_width, conf_width) + 2 * padding
        bg_height = text_height + conf_height + 3 * padding
        
        cv2.rectangle(frame, (20, 20), (20 + bg_width, 20 + bg_height), bg_color, -1)
        cv2.rectangle(frame, (20, 20), (20 + bg_width, 20 + bg_height), color, 2)
        
        # Draw text
        cv2.putText(frame, text, (20 + padding, 20 + padding + text_height), 
                   font, font_scale, color, thickness)
        cv2.putText(frame, conf_text, (20 + padding, 20 + padding + text_height + conf_height + 10), 
                   font, 1.0, color, 2)
        
        # Confidence bar
        bar_y = 20 + bg_height + 10
        bar_width = bg_width
        bar_height = 20
        
        # Background bar
        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Confidence bar
        conf_bar_width = int(bar_width * result['confidence'])
        cv2.rectangle(frame, (20, bar_y), (20 + conf_bar_width, bar_y + bar_height), color, -1)
        
        # Threshold line
        thresh_x = int(20 + bar_width * self.confidence_threshold)
        cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_height), (255, 255, 255), 2)
    
    def _draw_top_k_predictions(self, frame: np.ndarray, top_k_preds: List[Dict]):
        """Draw top-k predictions."""
        height, width = frame.shape[:2]
        
        # Position on the right side
        x_start = width - 250
        y_start = 20
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Background
        bg_height = len(top_k_preds) * 25 + 40
        cv2.rectangle(frame, (x_start - 10, y_start), (width - 10, y_start + bg_height), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "Top Predictions:", (x_start, y_start + 20), font, 0.7, (255, 255, 255), 1)
        
        # Predictions
        for i, pred in enumerate(top_k_preds):
            y = y_start + 40 + i * 25
            text = f"{pred['class_name'][:10]}: {pred['probability']:.2f}"
            
            # Color based on rank
            if i == 0:
                color = (0, 255, 0)  # Green for top prediction
            elif i == 1:
                color = (0, 255, 255)  # Yellow for second
            else:
                color = (255, 255, 255)  # White for others
            
            cv2.putText(frame, text, (x_start, y), font, font_scale, color, thickness)
    
    def _draw_performance_info(self, frame: np.ndarray):
        """Draw performance information."""
        height, width = frame.shape[:2]
        
        stats = self.inference_engine.get_performance_stats()
        if not stats:
            return
        
        # Position at bottom left
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        
        info_texts = [
            f"FPS: {stats['fps']:.1f}",
            f"Inference: {stats['avg_inference_time']*1000:.1f}ms",
            f"Frames: {stats['total_frames']}"
        ]
        
        y_start = height - 80
        for i, text in enumerate(info_texts):
            y = y_start + i * 20
            cv2.putText(frame, text, (10, y), font, font_scale, color, thickness)
    
    def _draw_controls_info(self, frame: np.ndarray):
        """Draw controls information."""
        height, width = frame.shape[:2]
        
        controls = [
            "SPACE: Reset smoothing",
            "R: Toggle recording",
            "D: Toggle debug info",
            "Q: Quit"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (200, 200, 200)
        
        x_start = width - 180
        y_start = height - len(controls) * 15 - 10
        
        for i, text in enumerate(controls):
            y = y_start + i * 15
            cv2.putText(frame, text, (x_start, y), font, font_scale, color, thickness)
    
    def _draw_recording_indicator(self, frame: np.ndarray):
        """Draw recording indicator."""
        height, width = frame.shape[:2]
        
        # Blinking red circle
        if int(time.time() * 2) % 2:  # Blink every 0.5 seconds
            cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
        
        cv2.putText(frame, "REC", (width - 55, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    def run(self):
        """Run the webcam demo."""
        print("\nStarting webcam demo...")
        print("Controls:")
        print("  SPACE: Reset prediction smoothing")
        print("  R: Toggle recording")
        print("  D: Toggle debug information")
        print("  Q: Quit")
        print("\nPress any key to start...")
        cv2.waitKey(0)
        
        self.is_running = True
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_thread)
        inference_thread = threading.Thread(target=self.inference_thread)
        
        capture_thread.start()
        inference_thread.start()
        
        try:
            # Main display loop
            while self.is_running:
                try:
                    # Get latest prediction result
                    frame, result = self.prediction_queue.get(timeout=0.1)
                    
                    # Draw UI
                    display_frame = self.draw_ui(frame, result)
                    
                    # Resize for display
                    if display_frame.shape[:2] != (self.display_size[1], self.display_size[0]):
                        display_frame = cv2.resize(display_frame, self.display_size)
                    
                    # Show frame
                    cv2.imshow('Sign Language Detection Demo', display_frame)
                    
                    # Handle keypresses
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_keypress(key)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Display error: {e}")
                    continue
        
        finally:
            # Cleanup
            self.is_running = False
            capture_thread.join(timeout=1)
            inference_thread.join(timeout=1)
            
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Save recording if any
            if self.prediction_history:
                self._save_recording()
            
            print("Demo finished")
    
    def _handle_keypress(self, key):
        """Handle keyboard input."""
        if key == ord('q') or key == ord('Q'):
            self.is_running = False
        elif key == ord(' '):  # Space
            self.inference_engine.reset_smoothing()
            print("Prediction smoothing reset")
        elif key == ord('r') or key == ord('R'):
            self.recording = not self.recording
            if self.recording:
                self.prediction_history = []
                print("Recording started")
            else:
                print("Recording stopped")
        elif key == ord('d') or key == ord('D'):
            self.show_debug_info = not self.show_debug_info
            print(f"Debug info: {'ON' if self.show_debug_info else 'OFF'}")
    
    def _save_recording(self):
        """Save recorded predictions."""
        if not self.prediction_history:
            return
        
        timestamp = int(time.time())
        filename = f"recording_{timestamp}.json"
        
        recording_data = {
            'timestamp': timestamp,
            'duration': self.prediction_history[-1]['timestamp'] - self.prediction_history[0]['timestamp'],
            'total_predictions': len(self.prediction_history),
            'predictions': self.prediction_history
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(recording_data, f, indent=2)
            print(f"Recording saved to {filename}")
        except Exception as e:
            print(f"Error saving recording: {e}")


def main():
    """Main function for running webcam demo from command line."""
    parser = argparse.ArgumentParser(description='Sign Language Detection Webcam Demo')
    
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--smoothing', type=int, default=5, help='Smoothing window size (default: 5)')
    parser.add_argument('--width', type=int, default=800, help='Display width (default: 800)')
    parser.add_argument('--height', type=int, default=600, help='Display height (default: 600)')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        global config
        config = Config(args.config)
    
    # Get class names
    dataset_name = config.get('dataset.asl_alphabet.path', 'asl_alphabet')
    if 'asl_alphabet' in dataset_name.lower():
        class_names = [
            "A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "nothing", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"
        ]
    else:
        # Try to get from config or use default
        class_names = config.get(f'dataset.{dataset_name}.classes', [str(i) for i in range(29)])
    
    # Create and run demo
    demo = WebcamDemo(
        model_path=args.model,
        class_names=class_names,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        smoothing_window=args.smoothing,
        display_size=(args.width, args.height)
    )
    
    try:
        demo.run()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    main()
