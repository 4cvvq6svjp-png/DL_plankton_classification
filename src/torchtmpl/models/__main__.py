# coding: utf-8

# Standard imports
import logging

# External imports
import torch

# Local imports
from . import build_model


def useless_function():
    logging.info(
        "This is a useless function, just to show you how to invoke the functions defined in the models/__main__.py script"
    )


def test_linear():
    cfg = {"class": "Linear"}
    input_size = (3, 128, 128)
    batch_size = 16
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # TODO
    # Fill in the expected output size
    expected_output_size = None
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    assert expected_output_size == output.shape
    print(f"Output tensor of size : {output.shape}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    useless_function()
    test_linear()
