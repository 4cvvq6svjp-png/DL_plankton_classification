import torch
from torchvision.transforms import v2

def get_transforms(split="train", img_size=224, aug_type="classic"):
    # 1. Base (Toujours appliqué)
    base_setup = [
        v2.ToImage(),
        v2.Resize((img_size, img_size)),
    ]
    
    augmentations = []
    
    # 2. Augmentations (Seulement pour le train)
    if split == "train":
        if aug_type == "heavy":
            augmentations = [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=360),
                v2.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        elif aug_type == "classic":
            # Augmentation très légère standard
            augmentations = [
                v2.RandomHorizontalFlip(p=0.5),
            ]

    # 3. Finalisation (Toujours à la fin)
    final_setup = [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    pipeline = base_setup + augmentations + final_setup
    return v2.Compose(pipeline)