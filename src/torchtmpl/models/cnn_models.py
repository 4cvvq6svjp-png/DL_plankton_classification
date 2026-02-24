# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def conv_down(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]




class FancyCNN(nn.Module):
    """
    A fancy CNN model with :
        - stacked 3x3 convolutions
        - convolutive down sampling
        - a global average pooling at the end
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        layers = []
        cin = input_size[0]
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # TODO: Implement the model
        self.model = nn.Sequential(*layers)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def forward(self, x):
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # TODO: Implement the forward pass
        return x
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    

import torch.nn as nn
from transformers import AutoModelForImageClassification

class HfModel(nn.Module):
    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        
        # 1. On récupère les infos de ton YAML (cfg correspond déjà à config["model"])
        model_name = cfg.get("name")
        freeze = cfg.get("freeze_backbone", True)
        
        # 2. Le cœur de la correction : AutoModelForImageClassification
        # Il s'adaptera automatiquement à EfficientNet, ResNet, ViT, etc.
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # 3. Le gel des poids (Transfer Learning)
        if freeze:
            # L'astuce magique : .base_model cible l'extracteur de features
            # de n'importe quel modèle, peu importe son créateur.
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            
    def forward(self, x):
        # Hugging Face attend un argument nommé 'pixel_values' pour les images
        outputs = self.model(pixel_values=x)
        return outputs.logits
