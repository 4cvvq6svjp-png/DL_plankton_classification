import torch
from torchvision.transforms import v2


def get_transforms(split="train", img_size=224, resize_size=256):
    base_setup = [
        v2.ToImage(),
        v2.Resize(resize_size),
    ]

    if split == "train":
        augmentations = [
            v2.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=90),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
        final_setup = [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    else:
        augmentations = [
            v2.CenterCrop(img_size),
        ]
        final_setup = [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    pipeline = base_setup + augmentations + final_setup
    return v2.Compose(pipeline)