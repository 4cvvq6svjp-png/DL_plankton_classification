# coding: utf-8

# External imports
import torch

# Local imports
from .base_models import *
from .cnn_models import *
# import the custom ResNet with dropout so that the simple builder can
# instantiate it when the user specifies ``class: ResNetDropout`` in the
# configuration.  We import the module here to keep names visible in the
# package namespace.
from .. import resnet_dropout  # noqa: F401
# bring the class into the models.* namespace so ``build_model``
# can instantiate it directly via ``cfg['class']``.
from .resnet_dropout import ResNetDropout


def build_model(cfg, input_size, num_classes):
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")
