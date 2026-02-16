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
    layers = []
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
    layers = []
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return nn.Sequential(*layers)
