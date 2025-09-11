#!/usr/bin/env python3
"""
Demo script for sign language detection.

Example usage:
    python demo.py webcam --model checkpoints/best_model.pth
    python demo.py streamlit --model model.pth
    python demo.py image --model model.pth --image test.jpg
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.demo.webcam_demo import WebcamDemo, main as webcam_main
from src.demo.streamlit_app import create_streamlit_app
from src.demo.inference_engine import InferenceEngine
from src.models.model_factory import create_model
from src.utils.config import config
import torch
import cv2
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sign Language Detection Demo')
    
    subparsers = parser.add_subparsers(dest='mode', help='Demo mode')
    
    # Webcam demo
    webcam_parser = subparsers.add_parser('webcam', help='Webcam demo')
    webcam_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    webcam_parser.add_argument('--camera', type=int, default=0, help='Camera index')
    webcam_parser.add_argument('--confidence', type=float, default=0.5, 
                              help='Confidence threshold')
    webcam_parser.add_argument('--smoothing', type=int, default=5,
                              help='Temporal smoothing window')
    webcam_parser.add_argument('--width', type=int, default=800, help='Display width')
    webcam_parser.add_argument('--height', type=int, default=600, help='Display height')
    
    # Streamlit app
    streamlit_parser = subparsers.add_parser('streamlit', help='Streamlit web app')
    
    # Image demo
    image_parser = subparsers.add_parser('image', help='Single image demo')
    image_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    image_parser.add_argument('--image', required=True, help='Path to input image')
    image_parser.add_argument('--output', help='Path to save output image')
    image_parser.add_argument('--top-k', type=int, default=5, help='Show top-k predictions')
    
    # Benchmark demo
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model performance')
    benchmark_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    benchmark_parser.add_argument('--iterations', type=int, default=100, 
                                 help='Number of benchmark iterations')
    benchmark_parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    benchmark_parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                                 help='Input image size (height width)')
    
    return parser.parse_args()


def load_model_and_classes(model_path: str):
    """Load model and get class names."""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get class names - try different sources
    class_names = None
    
    # Try from checkpoint metadata
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
    elif 'model_info' in checkpoint and 'class_names' in checkpoint['model_info']:
        class_names = checkpoint['model_info']['class_names']
    
    # Default to ASL alphabet classes
    if class_names is None:
        class_names = [
            "A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "nothing", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"
        ]
        print("Using default ASL alphabet classes")
    
    # Create model
    architecture = config.get('model.architecture', 'efficientnet_b0')
    model = create_model(
        architecture=architecture,
        num_classes=len(class_names),
        pretrained=False
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded: {model.__class__.__name__} with {len(class_names)} classes")
    return model, class_names


def webcam_demo(args):
    """Run webcam demo."""
    model, class_names = load_model_and_classes(args.model)
    
    demo = WebcamDemo(
        model_path=args.model,
        class_names=class_names,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        smoothing_window=args.smoothing,
        display_size=(args.width, args.height)
    )
    
    demo.run()


def streamlit_demo():
    """Run Streamlit web app."""
    print("Starting Streamlit app...")
    print("Open your browser to http://localhost:8501")
    
    import subprocess
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', 
        str(Path(__file__).parent / 'src' / 'demo' / 'streamlit_app.py')
    ])


def image_demo(args):
    """Run single image demo."""
    import matplotlib.pyplot as plt
    
    model, class_names = load_model_and_classes(args.model)
    
    # Create inference engine
    engine = InferenceEngine(model, class_names)
    
    # Load and process image
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing image: {image_path}")
    print(f"Image shape: {image.shape}")
    
    # Run inference
    result = engine.predict_single(image)
    
    # Get top-k predictions
    top_k_preds = engine.get_top_k_predictions(image, k=args.top_k)
    
    # Print results
    print(f"\nPrediction Results:")
    print(f"  Predicted class: {result['predicted_class_name']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Inference time: {result['inference_time']*1000:.2f} ms")
    
    print(f"\nTop-{args.top_k} Predictions:")
    for i, pred in enumerate(top_k_preds, 1):
        print(f"  {i}. {pred['class_name']}: {pred['probability']:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(f"Input Image\nPredicted: {result['predicted_class_name']} ({result['confidence']:.3f})")
    plt.axis('off')
    
    # Top-k predictions bar chart
    plt.subplot(1, 3, 2)
    classes = [pred['class_name'] for pred in top_k_preds]
    probs = [pred['probability'] for pred in top_k_preds]
    
    bars = plt.bar(classes, probs)
    bars[0].set_color('green')  # Highlight top prediction
    plt.title(f'Top-{args.top_k} Predictions')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    # All class probabilities (top 15)
    plt.subplot(1, 3, 3)
    all_probs = result['probabilities']
    top_15_indices = np.argsort(all_probs)[-15:][::-1]
    top_15_classes = [class_names[i] for i in top_15_indices]
    top_15_probs = [all_probs[i] for i in top_15_indices]
    
    plt.bar(top_15_classes, top_15_probs, alpha=0.7)
    plt.title('All Class Probabilities (Top 15)')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {args.output}")
    else:
        plt.show()


def benchmark_demo(args):
    """Run benchmark demo."""
    model, class_names = load_model_and_classes(args.model)
    
    # Create inference engine
    engine = InferenceEngine(model, class_names)
    
    print(f"Benchmarking model performance...")
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Device: {engine.device}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}")
    
    # Run benchmark
    results = engine.benchmark(
        num_iterations=args.iterations,
        image_size=tuple(args.image_size)
    )
    
    # Print detailed results
    print(f"\nBenchmark Results:")
    print(f"  Average inference time: {results['avg_time_per_inference']*1000:.2f} ms")
    print(f"  Min inference time: {results['min_time']*1000:.2f} ms")
    print(f"  Max inference time: {results['max_time']*1000:.2f} ms")
    print(f"  Std deviation: {results['std_time']*1000:.2f} ms")
    print(f"  Median time: {results['median_time']*1000:.2f} ms")
    print(f"  95th percentile: {results['p95_time']*1000:.2f} ms")
    print(f"  99th percentile: {results['p99_time']*1000:.2f} ms")
    print(f"  Average FPS: {results['avg_fps']:.1f}")
    print(f"  Total time: {results['total_time']:.2f} seconds")
    
    # Save results
    import json
    benchmark_file = f"benchmark_results_{model.__class__.__name__.lower()}.json"
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to: {benchmark_file}")


def main():
    """Main demo function."""
    args = parse_args()
    
    if args.mode is None:
        print("Please specify a demo mode: webcam, streamlit, image, or benchmark")
        print("Use --help for more information")
        return
    
    try:
        if args.mode == 'webcam':
            webcam_demo(args)
        elif args.mode == 'streamlit':
            streamlit_demo()
        elif args.mode == 'image':
            image_demo(args)
        elif args.mode == 'benchmark':
            benchmark_demo(args)
        else:
            print(f"Unknown demo mode: {args.mode}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        raise


if __name__ == '__main__':
    main()
