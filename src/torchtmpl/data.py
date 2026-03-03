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
import os
import matplotlib.pyplot as plt
from . import transforms as custom_transforms
from sklearn.model_selection import train_test_split


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
    valid_ratio = data_config.get("valid_ratio", 0.2)
    batch_size = data_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    root_dir = data_config["root_dir"]
    aug_type = data_config.get("augmentation", "classic")
    img_size = data_config.get("img_size", 224)
    resize_size = data_config.get("resize_size", 256)
    # --- LECTURE DE LA CONFIG ---
    aug_type = data_config.get("augmentation", "classic")

    logging.info(f"  - Setup: Augmentation={aug_type} | Stratified Split activé")

    train_path = os.path.join(root_dir, "train")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dossier introuvable: {train_path}")

    base_dataset = torchvision.datasets.ImageFolder(root=train_path)

    # --- NOUVEAU DÉCOUPAGE STRATIFIÉ ---
    indices = list(range(len(base_dataset)))
    targets = base_dataset.targets # On récupère toutes les étiquettes

    # train_test_split s'occupe de mélanger ET de respecter les proportions (stratify)
    train_indices, valid_indices = train_test_split(
        indices, 
        test_size=valid_ratio, 
        stratify=targets, 
        random_state=42 # Fixe l'aléatoire pour rendre tes tests reproductibles
    )

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)

    # --- APPLICATION DES TRANSFORMATIONS ---
    # Changement ici : on passe à 260 pour optimiser les performances de l'EfficientNet-B2
    train_transforms = custom_transforms.get_transforms(split="train", img_size=img_size, resize_size=resize_size)
    valid_transforms = custom_transforms.get_transforms(split="valid", img_size=img_size, resize_size=resize_size)

    train_dataset_wrapped = WrappedDataset(train_dataset, train_transforms)
    valid_dataset_wrapped = WrappedDataset(valid_dataset, valid_transforms)

    # --- WeightedRandomSampler pour équilibrer les classes rares ---
    train_targets = [base_dataset.targets[i] for i in train_indices]
    class_counts = np.bincount(train_targets)
    sample_weights = 1.0 / (class_counts[train_targets] + 1e-6)
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset_wrapped,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    num_classes = len(base_dataset.classes)
    input_size = tuple(train_dataset_wrapped[0][0].shape)
    
    # Calcul des poids de classes pour la CrossEntropyLoss
    class_weights = compute_class_weights(train_indices, base_dataset)

    return train_loader, valid_loader, input_size, num_classes, class_weights


def compute_class_weights(train_indices, base_dataset):
    """
    Compute class weights for handling imbalanced datasets.
    Weights are inversely proportional to class frequency.
    
    Arguments:
        train_indices: indices of training samples
        base_dataset: the base dataset with .targets attribute
    
    Returns:
        torch.Tensor: class weights (num_classes,)
    """
    train_targets = [base_dataset.targets[i] for i in train_indices]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / (class_counts + 1e-6)
    # Normalize so weights sum to num_classes
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    return torch.tensor(class_weights, dtype=torch.float32)



import sys
import yaml

def test_dataloaders(config_path):
    # On lit le vrai fichier de configuration passé en paramètre
    logging.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    data_config = config["data"]
    use_cuda = torch.cuda.is_available()

    try:
        train_loader, valid_loader, input_size, num_classes, class_weights = get_dataloaders(
            data_config, use_cuda
        )
        logging.info(f"Input size: {input_size}, Num classes: {num_classes}")

        X, y = next(iter(train_loader))
        logging.info(f"Batch loaded: X shape {X.shape}, y shape {y.shape}")
        
        # Sauvegarde et affichage
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




from PIL import Image

class KaggleTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Ordre trié pour reproductibilité et alignement avec l'ordre souvent attendu par Kaggle
        self.images = sorted(
            f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # On ouvre l'image en forçant le format RGB pour ResNet
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # On renvoie le tenseur de l'image ET son nom
        return image, img_name




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # On vérifie qu'on a bien donné le config.yaml dans la commande
    if len(sys.argv) != 2:
        logging.error(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(-1)
        
    test_dataloaders(sys.argv[1])
