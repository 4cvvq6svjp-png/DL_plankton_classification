# coding: utf-8

from functools import reduce
import operator

import torch.nn as nn


def Linear(cfg, input_size, num_classes):
    flattened_size = reduce(operator.mul, input_size, 1)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(flattened_size, num_classes)
    )


def FFN(cfg, input_size, num_classes):
    num_layers = cfg.get("num_layers", 1)
    num_hidden = cfg.get("num_hidden", 32)
    use_dropout = cfg.get("use_dropout", False)
    flattened_size = reduce(operator.mul, input_size, 1)
    layers = [nn.Flatten(), nn.Linear(flattened_size, num_hidden), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(num_hidden, num_hidden), nn.ReLU()])
        if use_dropout:
            layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(num_hidden, num_classes))
    return nn.Sequential(*layers)
    