import torch
from torchvision.transforms import v2

def get_transforms(split="train", img_size=260, resize_size=288):
    # 1. Le redimensionnement intelligent (Standard ImageNet)
    # En donnant un seul entier (256), PyTorch garde les proportions !
    base_setup = [
        v2.ToImage(),
        v2.Resize(resize_size), 
    ]
    
    # 2. Augmentations et Découpage
    augmentations = []
    if split == "train":
        augmentations = [
            v2.RandomRotation(degrees=30),
            v2.RandomCrop(img_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]
    else:
        # En test/validation, on prend toujours le carré de 224x224 bien au centre
        augmentations = [
            v2.CenterCrop(img_size)
        ]

    # 3. Finalisation (Les VRAIES couleurs attendues par Google)
    final_setup = [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    pipeline = base_setup + augmentations + final_setup
    return v2.Compose(pipeline)