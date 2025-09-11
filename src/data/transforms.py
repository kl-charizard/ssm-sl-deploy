"""Data transformation and augmentation utilities."""

import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any


class AlbumentationsTransform:
    """Wrapper for Albumentations transforms to work with PyTorch DataLoader."""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        transformed = self.transform(image=image)
        return transformed['image']


def get_transforms(
    input_size: Tuple[int, int] = (224, 224),
    mode: str = 'train',
    use_albumentations: bool = True
) -> transforms.Compose:
    """Get data transforms for training or inference.
    
    Args:
        input_size: Target image size (height, width)
        mode: 'train', 'val', or 'test'
        use_albumentations: Whether to use albumentations library
        
    Returns:
        Composed transforms
    """
    if use_albumentations:
        if mode == 'train':
            transform_list = [
                A.Resize(height=input_size[0], width=input_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ]
        else:
            transform_list = [
                A.Resize(height=input_size[0], width=input_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ]
        
        album_transform = A.Compose(transform_list)
        return AlbumentationsTransform(album_transform)
    
    else:
        # Traditional torchvision transforms
        if mode == 'train':
            transform_list = [
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        else:
            transform_list = [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        
        return transforms.Compose(transform_list)


def get_augmentation_transforms(config: Dict[str, Any]) -> A.Compose:
    """Get augmentation transforms based on configuration.
    
    Args:
        config: Augmentation configuration dictionary
        
    Returns:
        Albumentations compose object
    """
    transforms_list = []
    
    if config.get('rotation_range', 0) > 0:
        transforms_list.append(
            A.Rotate(
                limit=config['rotation_range'],
                p=0.5
            )
        )
    
    if config.get('width_shift_range', 0) > 0 or config.get('height_shift_range', 0) > 0:
        transforms_list.append(
            A.ShiftScaleRotate(
                shift_limit_x=config.get('width_shift_range', 0),
                shift_limit_y=config.get('height_shift_range', 0),
                scale_limit=0,
                rotate_limit=0,
                p=0.5
            )
        )
    
    if config.get('zoom_range', 0) > 0:
        transforms_list.append(
            A.RandomScale(
                scale_limit=config['zoom_range'],
                p=0.5
            )
        )
    
    if config.get('horizontal_flip', False):
        transforms_list.append(
            A.HorizontalFlip(p=0.5)
        )
    
    if config.get('brightness_range'):
        br_range = config['brightness_range']
        transforms_list.append(
            A.RandomBrightnessContrast(
                brightness_limit=(br_range[0] - 1.0, br_range[1] - 1.0),
                contrast_limit=0.1,
                p=0.5
            )
        )
    
    # Add some additional useful augmentations
    transforms_list.extend([
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),
    ])
    
    return A.Compose(transforms_list)


class TestTimeAugmentation:
    """Test Time Augmentation for improved inference accuracy."""
    
    def __init__(self, transforms: List[A.Compose], base_transform: A.Compose):
        """Initialize TTA.
        
        Args:
            transforms: List of augmentation transforms
            base_transform: Base transform without augmentation
        """
        self.transforms = transforms
        self.base_transform = base_transform
    
    def __call__(self, image: np.ndarray) -> List[torch.Tensor]:
        """Apply TTA transforms.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of transformed tensors
        """
        results = []
        
        # Base transform (no augmentation)
        base_result = self.base_transform(image=image)['image']
        results.append(base_result)
        
        # Apply each augmentation
        for transform in self.transforms:
            aug_result = transform(image=image)['image']
            results.append(aug_result)
        
        return results


def create_tta_transforms(input_size: Tuple[int, int] = (224, 224)) -> TestTimeAugmentation:
    """Create Test Time Augmentation transforms.
    
    Args:
        input_size: Target image size
        
    Returns:
        TTA transform object
    """
    # Base transform without augmentation
    base_transform = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Augmentation transforms for TTA
    tta_transforms = [
        A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Rotate(limit=5, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    ]
    
    return TestTimeAugmentation(tta_transforms, base_transform)
