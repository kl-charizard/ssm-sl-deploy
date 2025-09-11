#!/usr/bin/env python3
"""
Main training script for sign language detection models.

Example usage:
    python train.py --config config.yaml --dataset asl_alphabet
    python train.py --model efficientnet_b0 --epochs 50 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.data_loader import create_data_loaders
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.utils.config import config, Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train sign language detection model')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='asl_alphabet',
                       choices=['asl_alphabet', 'wlasl', 'custom'],
                       help='Dataset to use for training')
    parser.add_argument('--data-dir', type=str, help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model', type=str, help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--num-classes', type=int, help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd'], 
                       help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, 
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    
    # Regularization
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, help='Weight decay factor')
    parser.add_argument('--mixup-alpha', type=float, default=0.0, 
                       help='Mixup alpha parameter (0 = no mixup)')
    parser.add_argument('--cutmix-alpha', type=float, default=0.0,
                       help='CutMix alpha parameter (0 = no cutmix)')
    
    # System arguments
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                       default='auto', help='Device to use for training')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, help='Name for the experiment')
    parser.add_argument('--log-dir', type=str, help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, help='Directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=10, 
                       help='Save checkpoint every N epochs')
    parser.add_argument('--validate-every', type=int, default=1,
                       help='Run validation every N epochs')
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int,
                       help='Early stopping patience')
    
    # Quantization and deployment
    parser.add_argument('--quantize', action='store_true',
                       help='Apply quantization after training')
    parser.add_argument('--export-formats', nargs='+',
                       choices=['torchscript', 'onnx', 'tflite', 'coreml'],
                       help='Formats to export trained model to')
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration from arguments and config file."""
    # Load config file if provided
    if args.config:
        global config
        config = Config(args.config)
    
    # Override config with command line arguments
    if args.dataset:
        config.set('dataset.name', args.dataset)
    if args.data_dir:
        config.set(f'dataset.{args.dataset}.path', args.data_dir)
    if args.batch_size:
        config.set('training.batch_size', args.batch_size)
    if args.num_workers:
        config.set('system.num_workers', args.num_workers)
    
    if args.model:
        config.set('model.architecture', args.model)
    if args.pretrained is not None:
        config.set('model.pretrained', args.pretrained)
    if args.num_classes:
        config.set('model.num_classes', args.num_classes)
    if args.dropout:
        config.set('model.dropout_rate', args.dropout)
    
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.lr:
        config.set('training.learning_rate', args.lr)
    if args.optimizer:
        config.set('training.optimizer', args.optimizer)
    if args.scheduler:
        config.set('training.scheduler', args.scheduler)
    if args.weight_decay:
        config.set('training.weight_decay', args.weight_decay)
    
    if args.device != 'auto':
        config.set('system.device', args.device)
    if args.mixed_precision:
        config.set('system.mixed_precision', args.mixed_precision)
    
    if args.experiment_name:
        config.set('experiment.name', args.experiment_name)
    if args.log_dir:
        config.set('logging.log_dir', args.log_dir)
    if args.checkpoint_dir:
        config.set('logging.checkpoint_dir', args.checkpoint_dir)
    
    if args.early_stopping_patience:
        config.set('training.early_stopping_patience', args.early_stopping_patience)
    
    print("Configuration:")
    print(f"  Dataset: {config.get('dataset.name', args.dataset)}")
    print(f"  Model: {config.get('model.architecture')}")
    print(f"  Batch size: {config.get('training.batch_size')}")
    print(f"  Epochs: {config.get('training.epochs')}")
    print(f"  Learning rate: {config.get('training.learning_rate')}")
    print(f"  Device: {config.device}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup configuration
    setup_config(args)
    
    # Get dataset configuration
    dataset_name = config.get('dataset.name', args.dataset)
    data_dir = config.get(f'dataset.{dataset_name}.path')
    
    if not data_dir or not Path(data_dir).exists():
        raise ValueError(f"Dataset directory not found: {data_dir}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    data_loaders = create_data_loaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=config.get('training.batch_size'),
        num_workers=config.get('system.num_workers')
    )
    
    # Get class names
    if dataset_name == 'asl_alphabet':
        class_names = config.get('dataset.asl_alphabet.classes')
    else:
        # Try to get from first batch or use default
        class_names = [str(i) for i in range(config.get('model.num_classes', 29))]
    
    # Create model
    print(f"\nCreating model...")
    model = create_model(
        architecture=config.get('model.architecture'),
        num_classes=len(class_names),
        pretrained=config.get('model.pretrained'),
        dropout_rate=config.get('model.dropout_rate')
    )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders.get('test'),
        class_names=class_names,
        experiment_name=config.get('experiment.name', args.experiment_name)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Setup training configuration
    optimizer_config = {
        'name': config.get('training.optimizer'),
        'lr': config.get('training.learning_rate'),
        'weight_decay': config.get('training.weight_decay', 1e-4)
    }
    
    scheduler_config = {
        'name': config.get('training.scheduler'),
        'warmup_epochs': 5
    }
    
    loss_config = {
        'loss_type': 'cross_entropy'
    }
    
    early_stopping_config = None
    if config.get('training.early_stopping_patience'):
        early_stopping_config = {
            'patience': config.get('training.early_stopping_patience')
        }
    
    # Start training
    print(f"\nStarting training...")
    print("=" * 80)
    
    try:
        trainer.train(
            epochs=config.get('training.epochs'),
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            loss_config=loss_config,
            early_stopping_config=early_stopping_config,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            validate_every=args.validate_every
        )
        
        print("\nTraining completed successfully!")
        
        # Post-training tasks
        if args.quantize or args.export_formats:
            print("\nPost-training optimization...")
            
            # Quantization
            if args.quantize:
                from src.deployment.quantization import quantize_model
                print("Applying quantization...")
                quantized_model = quantize_model(
                    model, 
                    method='dynamic',
                    calibration_loader=data_loaders['val']
                )
                
                # Save quantized model
                checkpoint_dir = Path(config.get('logging.checkpoint_dir', 'checkpoints'))
                quantized_path = checkpoint_dir / trainer.experiment_name / 'quantized_model.pt'
                torch.save(quantized_model.state_dict(), quantized_path)
                print(f"Quantized model saved to {quantized_path}")
            
            # Model export
            if args.export_formats:
                from src.deployment.export import export_model
                print(f"Exporting model to formats: {args.export_formats}")
                
                export_dir = Path('exports') / trainer.experiment_name
                exported_models = export_model(
                    model=model,
                    class_names=class_names,
                    export_formats=args.export_formats,
                    output_dir=str(export_dir)
                )
                
                print("Exported models:")
                for fmt, path in exported_models.items():
                    print(f"  {fmt}: {path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
