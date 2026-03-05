# coding: utf-8

import logging

import torch

from . import build_model


def test_linear():
    cfg = {"class": "Linear"}
    input_size = (3, 128, 128)
    batch_size = 16
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)
    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    assert output.shape == torch.Size([batch_size, num_classes])
    print(f"Output tensor of size: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")


def test_ffn():
    logging.info("Testing FFN with 2 hidden layers")
    cfg = {"class": "FFN", "num_layers": 2, "num_hidden": 64, "use_dropout": False}
    input_size = (3, 128, 128)
    batch_size = 16
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)
    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    assert output.shape == torch.Size([batch_size, num_classes])
    print(f"Output tensor of size: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_linear()
    test_ffn()
