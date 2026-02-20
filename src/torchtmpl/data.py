# coding: utf-8

import logging
import random
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from pathlib import Path
from PIL import Image


# ==========================================================
# DATASET POUR INFERENCE (retourne image + filename)
# ==========================================================

class InferenceImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.samples = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )

        if len(self.samples) == 0:
            raise ValueError(f"No image found under test path: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = default_loader(str(path))

        if self.transform is not None:
            img = self.transform(img)

        return img, path.name


# ==========================================================
# TRANSFORM UTILE
# ==========================================================

class GrayToRGB(torch.nn.Module):
    def forward(self, img):
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
        return img


class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


# ==========================================================
# DATALOADERS
# ==========================================================

def get_dataloaders(data_config, use_cuda):

    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]

    logging.info("= Dataset creation")

    # -----------------------------
    # TRAIN DATASET
    # -----------------------------
    base_train_ds = ImageFolder(root=data_config["trainpath"])

    indices = list(range(len(base_train_ds)))
    random.shuffle(indices)

    num_valid = int(valid_ratio * len(base_train_ds))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    train_subset = torch.utils.data.Subset(base_train_ds, train_indices)
    valid_subset = torch.utils.data.Subset(base_train_ds, valid_indices)

    # -----------------------------
    # TRANSFORMS
    # -----------------------------
    preprocess_transforms = [
        v2.ToImage(),
        GrayToRGB(),
        v2.Resize(128),
        v2.CenterCrop(128),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ]

    train_transforms = v2.Compose(preprocess_transforms)
    valid_transforms = v2.Compose(preprocess_transforms)

    train_dataset = WrappedDataset(train_subset, train_transforms)
    valid_dataset = WrappedDataset(valid_subset, valid_transforms)

    # -----------------------------
    # TEST DATASET (IMPORTANT FIX)
    # -----------------------------
    test_dataset = InferenceImageDataset(
        root=data_config["testpath"],
        transform=valid_transforms
    )

    # -----------------------------
    # DATALOADERS
    # -----------------------------
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    num_classes = len(base_train_ds.classes)
    input_size = tuple(train_dataset[0][0].shape)

    return train_loader, valid_loader, test_loader, input_size, num_classes