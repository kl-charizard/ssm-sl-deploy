#!/usr/bin/env python3
"""
Custom dataset example for sign language detection.

This example shows how to:
1. Set up a custom dataset
2. Configure the framework for custom classes
3. Train a model on custom data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.dataset import SignLanguageDataset
from src.data.transforms import get_transforms
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from torch.utils.data import DataLoader


def create_custom_dataset_example():
    """Create an example custom dataset structure."""
    # Create example dataset structure
    dataset_dir = Path('examples/custom_dataset')
    
    # Define custom classes
    custom_classes = ['hello', 'thank_you', 'please', 'yes', 'no']
    
    print(f"Creating example custom dataset structure at {dataset_dir}")
    
    for class_name in custom_classes:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a README file explaining what images to put here
        readme_path = class_dir / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write(f"""
Place your '{class_name}' sign language images in this directory.

Requirements:
- Image formats: .jpg, .jpeg, .png
- Minimum images per class: 100 (recommended: 500+)
- Image quality: Clear, well-lit images
- Background: Varied backgrounds preferred
- Hand position: Consistent sign execution

Example filenames:
- {class_name}_001.jpg
- {class_name}_002.jpg
- etc.
""")
    
    # Create dataset configuration
    config_content = f"""
# Custom dataset configuration
custom_classes = {custom_classes}

# Training parameters
batch_size = 16
learning_rate = 0.001
epochs = 50

# Data splits
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
"""
    
    with open(dataset_dir / 'config.txt', 'w') as f:
        f.write(config_content)
    
    print(f"✅ Custom dataset structure created at {dataset_dir}")
    print(f"📝 Add your images to the class directories")
    print(f"⚙️  Configuration saved to {dataset_dir}/config.txt")
    
    return dataset_dir, custom_classes


def train_on_custom_dataset(dataset_dir: Path, custom_classes: list):
    """Train a model on custom dataset."""
    
    # Check if dataset has images
    total_images = 0
    for class_name in custom_classes:
        class_dir = dataset_dir / class_name
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        total_images += len(image_files)
        print(f"  {class_name}: {len(image_files)} images")
    
    if total_images < 50:  # Minimum for meaningful training
        print("❌ Not enough images found in custom dataset")
        print("Please add images to the class directories before training")
        return
    
    print(f"📊 Total images: {total_images}")
    
    # Create transforms
    train_transform = get_transforms(input_size=(224, 224), mode='train')
    val_transform = get_transforms(input_size=(224, 224), mode='val')
    
    # Create datasets
    train_dataset = SignLanguageDataset(
        data_dir=dataset_dir,
        split='train',
        transform=train_transform,
        classes=custom_classes,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    
    val_dataset = SignLanguageDataset(
        data_dir=dataset_dir,
        split='val',
        transform=val_transform,
        classes=custom_classes,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )
    
    print(f"📈 Training samples: {len(train_dataset)}")
    print(f"📉 Validation samples: {len(val_dataset)}")
    
    # Create model
    model = create_model(
        architecture='mobilenet_v3_small',  # Smaller model for custom data
        num_classes=len(custom_classes),
        pretrained=True,
        dropout_rate=0.3
    )
    
    print(f"🏗️  Model: {model.__class__.__name__}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=custom_classes,
        experiment_name='custom_dataset_example'
    )
    
    # Training configuration
    optimizer_config = {
        'name': 'adam',
        'lr': 0.0005,  # Lower learning rate for custom data
        'weight_decay': 1e-4
    }
    
    scheduler_config = {
        'name': 'step',
        'step_size': 10,
        'gamma': 0.5
    }
    
    loss_config = {
        'loss_type': 'cross_entropy'
    }
    
    early_stopping_config = {
        'patience': 8
    }
    
    # Start training
    print("\n🚀 Starting training on custom dataset...")
    print("=" * 50)
    
    try:
        trainer.train(
            epochs=20,  # Fewer epochs for custom data
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            loss_config=loss_config,
            early_stopping_config=early_stopping_config
        )
        
        print("\n✅ Custom dataset training completed!")
        
        # Save custom class mapping
        import json
        class_mapping_path = Path('examples/custom_class_mapping.json')
        with open(class_mapping_path, 'w') as f:
            json.dump({
                'classes': custom_classes,
                'class_to_idx': {cls: idx for idx, cls in enumerate(custom_classes)},
                'num_classes': len(custom_classes)
            }, f, indent=2)
        
        print(f"💾 Class mapping saved to {class_mapping_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def main():
    """Run custom dataset example."""
    print("🎯 Custom Dataset Example")
    print("=" * 50)
    
    # Create example dataset structure
    dataset_dir, custom_classes = create_custom_dataset_example()
    
    # Ask user if they want to proceed with training
    print("\n" + "=" * 50)
    print("📋 Next Steps:")
    print("1. Add your custom sign language images to the class directories")
    print("2. Ensure each class has at least 100 images")
    print("3. Run this script again to start training")
    print("\nWould you like to:")
    print("A) Continue with training (if you have images)")
    print("B) Skip training and just create the structure")
    
    choice = input("\nEnter choice (A/B): ").strip().upper()
    
    if choice == 'A':
        print("\n🚀 Proceeding with training...")
        train_on_custom_dataset(dataset_dir, custom_classes)
    else:
        print("\n📁 Dataset structure created. Add your images and run again!")
    
    print("\n🎉 Custom dataset example completed!")


if __name__ == '__main__':
    main()
