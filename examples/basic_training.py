#!/usr/bin/env python3
"""
Basic training example for sign language detection.

This example shows how to:
1. Set up a basic training pipeline
2. Train a model on ASL alphabet dataset
3. Save and evaluate the trained model
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from src.data.data_loader import create_data_loaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.config import config


def main():
    """Run basic training example."""
    print("🚀 Basic Training Example")
    print("=" * 50)
    
    # Configuration
    dataset_name = 'asl_alphabet'
    data_dir = 'datasets/asl_alphabet'
    model_architecture = 'efficientnet_b0'
    num_epochs = 10  # Short training for example
    
    # Check if dataset exists
    if not Path(data_dir).exists():
        print(f"❌ Dataset not found at {data_dir}")
        print("Please download the ASL alphabet dataset first.")
        print("See SETUP.md for instructions.")
        return
    
    print(f"📊 Dataset: {dataset_name}")
    print(f"🏗️  Model: {model_architecture}")
    print(f"📈 Epochs: {num_epochs}")
    print()
    
    # Create data loaders
    print("Loading data...")
    data_loaders = create_data_loaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=32,
        num_workers=4
    )
    
    # Get class names
    class_names = config.get('dataset.asl_alphabet.classes')
    
    # Create model
    print("Creating model...")
    model = create_model(
        architecture=model_architecture,
        num_classes=len(class_names),
        pretrained=True,
        dropout_rate=0.2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders.get('test'),
        class_names=class_names,
        experiment_name='basic_training_example'
    )
    
    # Training configuration
    optimizer_config = {
        'name': 'adam',
        'lr': 0.001,
        'weight_decay': 1e-4
    }
    
    scheduler_config = {
        'name': 'cosine',
        'warmup_epochs': 2
    }
    
    loss_config = {
        'loss_type': 'cross_entropy'
    }
    
    early_stopping_config = {
        'patience': 5
    }
    
    # Start training
    print("\nStarting training...")
    print("=" * 50)
    
    try:
        trainer.train(
            epochs=num_epochs,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            loss_config=loss_config,
            early_stopping_config=early_stopping_config,
            validate_every=1
        )
        
        print("\n✅ Training completed successfully!")
        
        # Evaluate on test set
        if 'test' in data_loaders:
            print("\nEvaluating on test set...")
            evaluator = ModelEvaluator(model, class_names)
            results = evaluator.evaluate_dataset(
                data_loaders['test'],
                save_predictions=True,
                save_dir='examples/results'
            )
            
            print(f"\nTest Results:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1-Score: {results['f1_macro']:.4f}")
            print(f"  Results saved to: examples/results")
        
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("- Check logs/ directory for training logs")
        print("- Check checkpoints/ directory for saved models")
        print("- Try running: python demo.py webcam --model checkpoints/basic_training_example/best_model.pth")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
