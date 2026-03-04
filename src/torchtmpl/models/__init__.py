# coding: utf-8

# External imports
import torch

# Local imports
# coding: utf-8

from .base_models import *
from .cnn_models import *
from .resnet_dropout import ResNetDropout


def build_model(cfg, input_size, num_classes):
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")