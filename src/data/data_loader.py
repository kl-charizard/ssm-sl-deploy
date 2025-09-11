"""Data loader utilities."""

from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import create_dataset
from .transforms import get_transforms
from ..utils.config import config


def create_data_loaders(
    dataset_name: str = 'asl_alphabet',
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    input_size: Optional[Tuple[int, int]] = None,
    use_weighted_sampling: bool = True,
    **kwargs
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        dataset_name: Name of the dataset to load
        data_dir: Path to dataset directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        input_size: Input image size (height, width)
        use_weighted_sampling: Whether to use weighted sampling for training
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    # Get configuration values
    if data_dir is None:
        data_dir = config.get(f'dataset.{dataset_name}.path')
    if batch_size is None:
        batch_size = config.get('training.batch_size', 32)
    if num_workers is None:
        num_workers = config.get('system.num_workers', 4)
    if input_size is None:
        input_size = tuple(config.get('model.input_size', [224, 224]))
    
    # Validate data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    
    # Get dataset splits configuration
    train_split = config.get(f'dataset.{dataset_name}.train_split', 0.8)
    val_split = config.get(f'dataset.{dataset_name}.val_split', 0.15)
    test_split = config.get(f'dataset.{dataset_name}.test_split', 0.05)
    
    # Create transforms
    train_transform = get_transforms(input_size=input_size, mode='train')
    val_transform = get_transforms(input_size=input_size, mode='val')
    test_transform = get_transforms(input_size=input_size, mode='test')
    
    # Create datasets
    train_dataset = create_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
        **kwargs
    )
    
    val_dataset = create_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
        **kwargs
    )
    
    test_dataset = create_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split='test',
        transform=test_transform,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
        **kwargs
    )
    
    # Print dataset information
    print(f"\nDataset: {dataset_name}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    
    # Print class distribution for training set
    print(f"\nTraining set class distribution:")
    train_distribution = train_dataset.get_class_distribution()
    for class_name, count in sorted(train_distribution.items()):
        print(f"  {class_name}: {count}")
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampling and len(train_dataset) > 0:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print("Using weighted random sampling for training")
    
    # Create data loaders
    pin_memory = config.get('system.pin_memory', True) and torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # Remove empty loaders
    data_loaders = {k: v for k, v in data_loaders.items() if len(v.dataset) > 0}
    
    return data_loaders


def get_class_names(dataset_name: str, data_dir: Optional[str] = None) -> list:
    """Get class names for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Path to dataset directory
        
    Returns:
        List of class names
    """
    if data_dir is None:
        data_dir = config.get(f'dataset.{dataset_name}.path')
    
    # Create a dummy dataset to get class names
    dataset = create_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split='train'
    )
    
    return dataset.classes


def calculate_dataset_stats(data_loader: DataLoader) -> Dict[str, float]:
    """Calculate mean and std statistics for a dataset.
    
    Args:
        data_loader: Data loader for the dataset
        
    Returns:
        Dictionary with mean and std for each channel
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    print("Calculating dataset statistics...")
    for images, _ in data_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist()
    }


def preview_dataset(data_loader: DataLoader, num_samples: int = 8) -> None:
    """Preview samples from a dataset (for debugging).
    
    Args:
        data_loader: Data loader to preview
        num_samples: Number of samples to show
    """
    import matplotlib.pyplot as plt
    
    # Get a batch
    images, labels = next(iter(data_loader))
    
    # Convert to displayable format
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        # Denormalize
        img = img * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy and transpose
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
