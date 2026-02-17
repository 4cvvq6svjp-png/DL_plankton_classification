# coding: utf-8

# Standard imports
import logging
import random

# External imports
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
import PIL

import matplotlib.pyplot as plt


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()


class GrayToRGB(torch.nn.Module):
    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be converted to RGB if necessary
                          if Tensor, expected to be (C, H, W)

        Returns:
            PIL Image or Tensor: RGB image
        """
        if isinstance(img, torch.Tensor):
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
        else:
            raise TypeError(f"Expected Tensor, got {type(img)}")
        return img


class WrappedDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        xi, yi = self.dataset[idx]
        t_xi = self.transform(xi)
        return t_xi, yi

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset}, transform={self.transform})"

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    root_dir = data_config["root_dir"]

    logging.info("  - Dataset creation")

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # TODO: Create the Caltech101 dataset
    #       The variable rootdir is useful
    base_dataset = torchvision.datasets.Caltech101(root=root_dir, download=False)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # TODO : Create the train and valid splits. The torch.utils.data.Subset
    #        class is useful for this purpose
    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    preprocess_transforms = [
        v2.ToImage(),
        GrayToRGB(),
        v2.Resize(128),  # Keeps the aspect ratio
        v2.RandomCrop(128),  # Crops the variable size to a fixed 128 x 128
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    augmentation_transforms = [
    ]

    train_transforms = v2.Compose(preprocess_transforms + augmentation_transforms)
    train_dataset = WrappedDataset(train_dataset, train_transforms)

    valid_transforms = v2.Compose(preprocess_transforms)
    valid_dataset = WrappedDataset(valid_dataset, valid_transforms)

    # Build the dataloaders
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # TODO: Create the train and valid dataloaders
    # from their respective datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    num_classes = len(base_dataset.categories)
    input_size = tuple(train_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes


def test_dataloaders():
    data_config = {
        "root_dir": "/mounts/datasets/datasets/",
        "valid_ratio": 0.2,
        "batch_size": 32,
        "num_workers": 0,
    }
    use_cuda = torch.cuda.is_available()

    try:
        train_loader, valid_loader, input_size, num_classes = get_dataloaders(
            data_config, use_cuda
        )
        logging.info(f"Input size: {input_size}, Num classes: {num_classes}")

        X, y = next(iter(train_loader))
        logging.info(f"Batch loaded: X shape {X.shape}, y shape {y.shape}")
        grid = make_grid(X, nrow=8)
        save_image(grid, "batch_preview.png")
        logging.info("Image saved to batch_preview.png")
        show(grid)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error in test_dataloaders: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataloaders()
