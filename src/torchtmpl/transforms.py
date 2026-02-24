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
                v2.RandomRotation(degrees=30),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                v2.RandomAffine(degrees=0, scale=(0.9, 1.1)),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ]
        elif aug_type == "classic":
            # Augmentation standard modérée pour les planctons
            augmentations = [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.3),
                v2.RandomRotation(degrees=20),
                v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            ]

    # 3. Finalisation (Toujours à la fin)
    # NOTE: Pour grayscale converti en RGB, les 3 canaux sont identiques
    # On utilise une normalisation adaptée au plancton
    final_setup = [
        v2.ToDtype(torch.float32, scale=True),
        # Normalisation adaptée aux données plancton (grayscale→RGB)
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    
    pipeline = base_setup + augmentations + final_setup
    return v2.Compose(pipeline)