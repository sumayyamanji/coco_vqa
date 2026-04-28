"""Image transforms for training and validation."""
from __future__ import annotations

from torchvision import transforms

# ImageNet statistics used by both CLIP and timm models
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Augmented pipeline used during training."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Deterministic pipeline used during validation and inference."""
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )
