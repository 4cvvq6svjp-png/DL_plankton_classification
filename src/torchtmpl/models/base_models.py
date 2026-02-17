# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch.nn as nn


def Linear(cfg, input_size, num_classes):
    """
    cfg: a dictionnary with possibly some parameters
    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    # TODO: Implement a simple linear model
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # Calculate flattened input size (C * H * W)
    flattened_size = reduce(operator.mul, input_size, 1)
    layers = [
        nn.Flatten(),
        nn.Linear(flattened_size, num_classes)
    ]
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    return nn.Sequential(*layers)


def FFN(cfg, input_size, num_classes):
    """
    cfg: a dictionnary with possibly some parameters
    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    num_layers = cfg.get("num_layers", 1)
    num_hidden = cfg.get("num_hidden", 32)
    use_dropout = cfg.get("use_dropout", False)
    # TODO: Implement a simple linear model
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # Calculate flattened input size (C * H * W)
    flattened_size = reduce(operator.mul, input_size, 1)
    layers = [nn.Flatten()]
    
    # First hidden layer
    layers.append(nn.Linear(flattened_size, num_hidden))
    layers.append(nn.ReLU())
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    
    # Additional hidden layers
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(num_hidden, num_hidden))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(0.5))
    
    # Output layer
    layers.append(nn.Linear(num_hidden, num_classes))
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return nn.Sequential(*layers)
    