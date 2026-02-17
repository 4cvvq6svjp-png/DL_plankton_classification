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
