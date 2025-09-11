#!/usr/bin/env python3
"""
Model evaluation script for sign language detection models.

Example usage:
    python evaluate.py --model-path checkpoints/best_model.pth --dataset asl_alphabet
    python evaluate.py --model-path model.pth --data-dir data/custom --tta
"""

import argparse
import sys
import torch
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.data_loader import create_data_loaders
from src.models.model_factory import create_model
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.analysis import ModelAnalysis
from src.utils.config import config, Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate sign language detection model')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-architecture', type=str, 
                       help='Model architecture (if not in checkpoint)')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='asl_alphabet',
                       choices=['asl_alphabet', 'wlasl', 'custom'],
                       help='Dataset to evaluate on')
    parser.add_argument('--data-dir', type=str, help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Evaluation options
    parser.add_argument('--tta', action='store_true',
                       help='Use test time augmentation')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save detailed predictions')
    parser.add_argument('--analysis', action='store_true',
                       help='Run detailed model analysis')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive evaluation report')
    
    # System arguments
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                       default='auto', help='Device to use for evaluation')
    
    return parser.parse_args()


def load_model(model_path: str, architecture: str = None, num_classes: int = 29):
    """Load model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Try to get architecture from checkpoint
    if 'model_info' in checkpoint:
        model_info = checkpoint['model_info']
        print(f"Model info: {model_info}")
    
    # Determine architecture
    if architecture is None:
        architecture = config.get('model.architecture', 'efficientnet_b0')
    
    # Create model
    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded: {model.__class__.__name__}")
    return model, checkpoint


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = config.device
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Get dataset configuration
    data_dir = args.data_dir or config.get(f'dataset.{args.dataset}.path')
    
    if not data_dir or not Path(data_dir).exists():
        raise ValueError(f"Dataset directory not found: {data_dir}")
    
    # Create data loaders
    print(f"\nLoading {args.dataset} dataset from {data_dir}...")
    data_loaders = create_data_loaders(
        dataset_name=args.dataset,
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get evaluation data loader
    if args.split not in data_loaders:
        raise ValueError(f"Split '{args.split}' not available in dataset")
    
    eval_loader = data_loaders[args.split]
    print(f"Evaluating on {args.split} split: {len(eval_loader.dataset)} samples")
    
    # Get class names
    if args.dataset == 'asl_alphabet':
        class_names = config.get('dataset.asl_alphabet.classes')
    else:
        # Try to get from dataset or use default
        class_names = [str(i) for i in range(29)]  # Default
    
    # Load model
    model, checkpoint = load_model(
        args.model_path, 
        args.model_architecture,
        len(class_names)
    )
    model.to(device)
    
    # Create evaluator
    print(f"\nInitializing evaluator...")
    evaluator = ModelEvaluator(model, class_names, device)
    
    # Run evaluation
    print(f"\nRunning evaluation...")
    print("=" * 60)
    
    results = evaluator.evaluate_dataset(
        eval_loader,
        use_tta=args.tta,
        save_predictions=args.save_predictions,
        save_dir=args.output_dir if args.save_predictions else None
    )
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"  Dataset: {args.dataset} ({args.split} split)")
    print(f"  Samples: {results['dataset_size']}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {results.get('top_5_accuracy', 'N/A'):.4f}")
    print(f"  F1-Score (Macro): {results['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"  Precision (Macro): {results['precision_macro']:.4f}")
    print(f"  Recall (Macro): {results['recall_macro']:.4f}")
    
    if 'expected_calibration_error' in results:
        print(f"  Expected Calibration Error: {results['expected_calibration_error']:.4f}")
    
    # Per-class results
    print(f"\nPer-Class Accuracy:")
    per_class_acc = evaluator.get_per_class_accuracy()
    for class_name, acc in sorted(per_class_acc.items()):
        print(f"  {class_name}: {acc:.4f}")
    
    # Detailed analysis
    if args.analysis:
        print(f"\nRunning detailed analysis...")
        analyzer = ModelAnalysis(model, class_names, device)
        
        # Feature space visualization
        print("Generating feature space visualization...")
        analyzer.visualize_feature_space(
            eval_loader,
            save_path=Path(args.output_dir) / 'feature_space.png'
        )
        
        # Gradient analysis
        print("Analyzing gradients...")
        grad_analysis = analyzer.analyze_gradients(eval_loader)
        print(f"Gradient analysis: {grad_analysis}")
    
    # Misclassification analysis
    if args.save_predictions:
        print(f"\nAnalyzing misclassifications...")
        miscl_analysis = evaluator.analyze_misclassifications()
        
        if 'top_error_patterns' in miscl_analysis:
            print(f"Top error patterns:")
            for pattern in miscl_analysis['top_error_patterns'][:5]:
                print(f"  {pattern['true_class']} → {pattern['predicted_class']}: "
                      f"{pattern['count']} errors ({pattern['percentage_of_errors']:.1f}%)")
        
        # Confidence analysis
        conf_analysis = evaluator.analyze_confidence_distribution()
        if conf_analysis:
            print(f"\nConfidence Analysis:")
            print(f"  Correct predictions mean confidence: "
                  f"{conf_analysis['correct_predictions']['mean_confidence']:.4f}")
            print(f"  Incorrect predictions mean confidence: "
                  f"{conf_analysis['incorrect_predictions']['mean_confidence']:.4f}")
    
    # Generate plots
    print(f"\nGenerating visualizations...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(
        save_path=str(output_dir / 'confusion_matrix.png')
    )
    
    # Confidence distribution
    if args.save_predictions:
        evaluator.plot_confidence_distribution(
            save_path=str(output_dir / 'confidence_distribution.png')
        )
    
    # Generate comprehensive report
    if args.generate_report:
        print(f"\nGenerating comprehensive report...")
        report_path = evaluator.generate_evaluation_report(args.output_dir)
        print(f"Report saved to: {report_path}")
    
    # Model benchmarking
    print(f"\nBenchmarking model performance...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    import time
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Average FPS: {1/avg_time:.1f}")
    
    # Save benchmark results
    benchmark_results = {
        'avg_inference_time_ms': avg_time * 1000,
        'fps': 1 / avg_time,
        'device': str(device),
        'batch_size': 1,
        'model_architecture': model.__class__.__name__
    }
    
    import json
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
