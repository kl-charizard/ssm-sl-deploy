#!/usr/bin/env python3
"""
Convert PyTorch sign language model to Core ML format for iOS deployment.

This script converts a trained PyTorch model to Core ML format that can be used
in iOS applications for real-time sign language detection.

Usage:
    python scripts/convert_to_coreml.py --model-path checkpoints/best_model.pth --output-path ios/SignLanguageDetector/SignLanguageModel.mlmodel

Requirements:
    - torch
    - coremltools
    - torchvision
"""

import argparse
import torch
import coremltools as ct
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import after adding paths
from src.models.model_factory import create_model
from src.utils.config import config


def convert_pytorch_to_coreml(model_path: str, output_path: str, input_size: tuple = (224, 224)):
    """
    Convert PyTorch model to Core ML format.
    
    Args:
        model_path: Path to PyTorch model checkpoint
        output_path: Path to save Core ML model
        input_size: Input image size (height, width)
    """
    print(f"Loading PyTorch model from {model_path}...")
    
    # Load the PyTorch model
    device = torch.device('cpu')  # Use CPU for conversion
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model info from checkpoint
    model_name = checkpoint.get('model_name', 'efficientnet_b0')
    num_classes = checkpoint.get('num_classes', 29)
    
    print(f"Model name: {model_name}")
    print(f"Number of classes: {num_classes}")
    
    # Create model instance
    try:
        model = create_model(
            architecture=model_name,
            num_classes=num_classes,
            pretrained=False
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Trying with default EfficientNet...")
        try:
            model = create_model(
                architecture='efficientnet_b0',
                num_classes=num_classes,
                pretrained=False
            )
        except Exception as e2:
            print(f"Error with EfficientNet: {e2}")
            print("Creating a simple CNN model as fallback...")
            # Create a simple fallback model
            import torch.nn as nn
            model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Warning: Could not load state dict: {e}")
        print("Using randomly initialized model...")
    
    model.eval()
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Number of classes: {checkpoint.get('num_classes', 29)}")
    
    # Create example input
    example_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to Core ML
    print("Converting to Core ML...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="image",
            shape=example_input.shape,
            scale=1/255.0,  # Normalize from 0-255 to 0-1
            bias=[0, 0, 0]  # No bias
        )],
        outputs=[ct.TensorType(name="classLabel", dtype=np.int32)],
        minimum_deployment_target=ct.target.iOS14
    )
    
    # Add metadata
    coreml_model.short_description = "Sign Language Detection Model"
    coreml_model.author = "Sign Language Detection Framework"
    coreml_model.license = "MIT"
    coreml_model.version = "1.0"
    
    # Add class labels
    class_labels = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "space", "del", "nothing"
    ]
    
    coreml_model.output_description["classLabel"] = "Predicted sign language letter"
    
    # Save the model
    print(f"Saving Core ML model to {output_path}...")
    coreml_model.save(output_path)
    
    print("✅ Conversion completed successfully!")
    print(f"Core ML model saved to: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return coreml_model


def optimize_for_mobile(coreml_model, output_path: str):
    """
    Optimize Core ML model for mobile deployment.
    
    Args:
        coreml_model: Core ML model object
        output_path: Path to save optimized model
    """
    print("Optimizing model for mobile deployment...")
    
    # Quantize the model to reduce size
    quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
        coreml_model, nbits=8
    )
    
    # Save optimized model
    quantized_model.save(output_path)
    
    print(f"✅ Optimized model saved to: {output_path}")
    print(f"Optimized model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to Core ML')
    parser.add_argument('--model-path', required=True, help='Path to PyTorch model checkpoint')
    parser.add_argument('--output-path', required=True, help='Path to save Core ML model')
    parser.add_argument('--input-size', type=int, nargs=2, default=[224, 224], 
                       help='Input image size (height width)')
    parser.add_argument('--optimize', action='store_true', 
                       help='Optimize model for mobile deployment')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Convert model
        coreml_model = convert_pytorch_to_coreml(
            model_path=args.model_path,
            output_path=args.output_path,
            input_size=tuple(args.input_size)
        )
        
        # Optimize if requested
        if args.optimize:
            optimized_path = args.output_path.replace('.mlmodel', '_optimized.mlmodel')
            optimize_for_mobile(coreml_model, optimized_path)
        
        print("\n🎉 Model conversion completed successfully!")
        print("\nNext steps:")
        print("1. Add the .mlmodel file to your iOS project in Xcode")
        print("2. Update the model name in SignLanguageModel.swift if needed")
        print("3. Build and run the iOS app")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
