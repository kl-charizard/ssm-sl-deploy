"""Hand detection module using MediaPipe for sign language detection."""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List, Dict, Any
import time


class HandDetector:
    """Hand detection using MediaPipe for real-time sign language detection."""
    
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        crop_padding: float = 0.2
    ):
        """Initialize hand detector.
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            crop_padding: Extra padding around detected hand (as fraction of bounding box)
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.crop_padding = crop_padding
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Performance tracking
        self.detection_times = []
        self.last_detection_time = 0
        
        print(f"Hand detector initialized (max_hands={max_num_hands}, confidence={min_detection_confidence})")
    
    def detect_hands(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect hands in the given image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of hand detection results, each containing:
            - 'landmarks': Hand landmarks
            - 'bounding_box': Bounding box (x, y, w, h)
            - 'confidence': Detection confidence
            - 'handedness': Left or right hand
        """
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(rgb_image)
        
        hand_detections = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand classification (left/right)
                handedness = "Unknown"
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                
                # Calculate bounding box
                h, w, _ = image.shape
                landmarks_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
                
                x_min, y_min = landmarks_array.min(axis=0).astype(int)
                x_max, y_max = landmarks_array.max(axis=0).astype(int)
                
                # Add padding
                padding_x = int((x_max - x_min) * self.crop_padding)
                padding_y = int((y_max - y_min) * self.crop_padding)
                
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(w, x_max + padding_x)
                y_max = min(h, y_max + padding_y)
                
                hand_detections.append({
                    'landmarks': hand_landmarks,
                    'bounding_box': (x_min, y_min, x_max - x_min, y_max - y_min),
                    'confidence': 1.0,  # MediaPipe doesn't provide per-hand confidence
                    'handedness': handedness
                })
        
        # Track performance
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 100:
            self.detection_times.pop(0)
        
        self.last_detection_time = detection_time
        
        return hand_detections
    
    def crop_hand_region(self, image: np.ndarray, hand_detection: Dict[str, Any]) -> Optional[np.ndarray]:
        """Crop hand region from image based on detection.
        
        Args:
            image: Input image
            hand_detection: Hand detection result
            
        Returns:
            Cropped hand image or None if invalid detection
        """
        x, y, w, h = hand_detection['bounding_box']
        
        # Ensure valid bounding box
        if w <= 0 or h <= 0:
            return None
        
        # Crop the hand region
        cropped = image[y:y+h, x:x+w]
        
        # Ensure we have a valid crop
        if cropped.size == 0:
            return None
        
        return cropped
    
    def get_best_hand(self, hand_detections: List[Dict[str, Any]], 
                      preferred_hand: str = "right") -> Optional[Dict[str, Any]]:
        """Get the best hand detection for sign language recognition.
        
        Args:
            hand_detections: List of hand detections
            preferred_hand: Preferred hand ("left" or "right")
            
        Returns:
            Best hand detection or None
        """
        if not hand_detections:
            return None
        
        # If only one hand, return it
        if len(hand_detections) == 1:
            return hand_detections[0]
        
        # Prefer the specified hand
        for detection in hand_detections:
            if detection['handedness'].lower() == preferred_hand.lower():
                return detection
        
        # If preferred hand not found, return the first one
        return hand_detections[0]
    
    def draw_hand_landmarks(self, image: np.ndarray, hand_detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw hand landmarks and bounding boxes on image.
        
        Args:
            image: Input image
            hand_detections: List of hand detections
            
        Returns:
            Image with drawn landmarks and bounding boxes
        """
        output_image = image.copy()
        
        for detection in hand_detections:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                output_image,
                detection['landmarks'],
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw bounding box
            x, y, w, h = detection['bounding_box']
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw hand label
            label = f"{detection['handedness']} Hand"
            cv2.putText(output_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output_image
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.detection_times:
            return {'avg_detection_time': 0.0, 'fps': 0.0}
        
        avg_time = np.mean(self.detection_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            'avg_detection_time': avg_time,
            'fps': fps,
            'last_detection_time': self.last_detection_time
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
