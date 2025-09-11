"""Dataset classes for sign language detection."""

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


class SignLanguageDataset(Dataset):
    """Generic sign language dataset class."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[Any] = None,
        classes: Optional[List[str]] = None,
        use_cache: bool = True,
        cache_suffix: str = '',
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        random_state: int = 42
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Data transforms to apply
            classes: List of class names. If None, inferred from directories
            use_cache: Whether to use cached data if available
            cache_suffix: Suffix for cache file names
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split  
            test_ratio: Ratio for test split
            random_state: Random state for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_cache = use_cache
        self.cache_suffix = cache_suffix
        self.random_state = random_state
        
        # Validate split ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Get classes
        if classes is None:
            self.classes = self._discover_classes()
        else:
            self.classes = classes
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load or create dataset
        self.samples = self._load_samples()
        
    def _discover_classes(self) -> List[str]:
        """Discover class names from directory structure."""
        classes = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                classes.append(item.name)
        
        return sorted(classes)
    
    def _get_cache_path(self, split: str) -> Path:
        """Get cache file path for a split."""
        cache_name = f"cached_{split}"
        if self.cache_suffix:
            cache_name += f"_{self.cache_suffix}"
        cache_name += ".pkl"
        
        return self.data_dir / cache_name
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load dataset samples."""
        cache_path = self._get_cache_path(self.split)
        
        # Try to load from cache
        if self.use_cache and cache_path.exists():
            try:
                print(f"Loading cached data from {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    return cached_data['samples']
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}")
        
        # Collect all samples
        all_samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} does not exist")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(class_dir.glob(f"*{ext}")))
                image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
            
            for img_path in image_files:
                all_samples.append((img_path, class_idx))
        
        print(f"Found {len(all_samples)} total samples across {len(self.classes)} classes")
        
        # Split data
        train_samples, temp_samples = train_test_split(
            all_samples,
            test_size=(1 - self.train_ratio),
            random_state=self.random_state,
            stratify=[sample[1] for sample in all_samples]
        )
        
        if self.val_ratio > 0 and self.test_ratio > 0:
            val_test_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_samples, test_samples = train_test_split(
                temp_samples,
                test_size=(1 - val_test_ratio),
                random_state=self.random_state,
                stratify=[sample[1] for sample in temp_samples]
            )
        elif self.val_ratio > 0:
            val_samples = temp_samples
            test_samples = []
        else:
            val_samples = []
            test_samples = temp_samples
        
        # Cache splits
        splits_data = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        if self.use_cache:
            for split_name, split_samples in splits_data.items():
                if split_samples:  # Only cache non-empty splits
                    split_cache_path = self._get_cache_path(split_name)
                    try:
                        with open(split_cache_path, 'wb') as f:
                            pickle.dump({
                                'samples': split_samples,
                                'classes': self.classes,
                                'class_to_idx': self.class_to_idx
                            }, f)
                        print(f"Cached {split_name} split to {split_cache_path}")
                    except Exception as e:
                        print(f"Failed to cache {split_name} split: {e}")
        
        # Return requested split
        split_samples = splits_data[self.split]
        print(f"Loaded {len(split_samples)} samples for {self.split} split")
        
        return split_samples
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, class_index)
        """
        img_path, class_idx = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for the current split."""
        distribution = {}
        for _, class_idx in self.samples:
            class_name = self.idx_to_class[class_idx]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        
        return distribution
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced sampling."""
        class_counts = {}
        for _, class_idx in self.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        # Calculate inverse frequency weights
        total_samples = len(self.samples)
        weights = []
        
        for _, class_idx in self.samples:
            weight = total_samples / (len(class_counts) * class_counts[class_idx])
            weights.append(weight)
        
        return torch.FloatTensor(weights)


class ASLAlphabetDataset(SignLanguageDataset):
    """Specific dataset class for ASL Alphabet dataset."""
    
    def __init__(self, data_dir: Union[str, Path], **kwargs):
        """Initialize ASL Alphabet dataset.
        
        Args:
            data_dir: Path to ASL alphabet dataset directory
            **kwargs: Additional arguments for parent class
        """
        # ASL alphabet classes
        asl_classes = [
            "A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "nothing", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"
        ]
        
        super().__init__(data_dir=data_dir, classes=asl_classes, **kwargs)


class WSALDataset(SignLanguageDataset):
    """Dataset class for Word-Level American Sign Language (WLASL) dataset."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        json_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize WLASL dataset.
        
        Args:
            data_dir: Path to WLASL dataset directory
            json_file: Path to WLASL JSON metadata file
            **kwargs: Additional arguments for parent class
        """
        self.json_file = json_file
        
        # WLASL is video-based, so we'll need different handling
        # This is a placeholder for future implementation
        super().__init__(data_dir=data_dir, **kwargs)
        
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load WLASL samples (to be implemented)."""
        # TODO: Implement WLASL dataset loading
        # This would involve loading video files and extracting frames
        raise NotImplementedError("WLASL dataset loading not yet implemented")


def create_dataset(
    dataset_name: str,
    data_dir: Union[str, Path],
    split: str = 'train',
    **kwargs
) -> SignLanguageDataset:
    """Factory function to create appropriate dataset.
    
    Args:
        dataset_name: Name of the dataset ('asl_alphabet', 'wlasl', 'custom')
        data_dir: Path to dataset directory
        split: Dataset split
        **kwargs: Additional arguments for dataset
        
    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'asl_alphabet':
        return ASLAlphabetDataset(data_dir=data_dir, split=split, **kwargs)
    elif dataset_name == 'wlasl':
        return WSALDataset(data_dir=data_dir, split=split, **kwargs)
    elif dataset_name == 'custom':
        return SignLanguageDataset(data_dir=data_dir, split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
