# coding: utf-8

from functools import reduce
import operator

import torch
import torch.nn as nn
from torchvision import models as tv_models
from transformers import AutoModelForImageClassification

class HfModel(nn.Module):
    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        model_name = cfg.get("name")
        freeze = cfg.get("freeze_backbone", True)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        if freeze:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits


class TorchVisionResNet(nn.Module):
    """
    ResNet50 torchvision (conv1, layer1-4, fc) pour charger les checkpoints
    entraînés avec l'API torchvision (p.ex. d'un collègue), dont les clés
    state_dict sont de la forme model.conv1.weight, model.layer1.*, model.fc.*.
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        model_name = (cfg.get("name") or "resnet50").lower()
        if "resnet50" in model_name or model_name == "resnet50":
            backbone = tv_models.resnet50(weights=None)
        else:
            raise ValueError(f"TorchVisionResNet: seul resnet50 est supporté, reçu name={model_name}")
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def _replace_tv_classifier(backbone, num_classes, classifier_attr="classifier"):
    """Remplace la tête de classification (EfficientNet, ConvNeXt)."""
    head = getattr(backbone, classifier_attr)
    if isinstance(head, nn.Sequential):
        for i in range(len(head) - 1, -1, -1):
            if isinstance(head[i], nn.Linear):
                in_features = head[i].in_features
                new_head = list(head)
                new_head[i] = nn.Linear(in_features, num_classes)
                setattr(backbone, classifier_attr, nn.Sequential(*new_head))
                return
    raise ValueError(f"Impossible de trouver un Linear dans {classifier_attr}")


class TorchVisionEfficientNet(nn.Module):
    """
    EfficientNet torchvision (B0–B7) pour entraînement / prédiction.
    state_dict : model.features.*, model.avgpool, model.classifier.*
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        name = (cfg.get("name") or "efficientnet_b0").lower()
        builders = {
            "efficientnet_b0": tv_models.efficientnet_b0,
            "efficientnet_b1": tv_models.efficientnet_b1,
            "efficientnet_b2": tv_models.efficientnet_b2,
            "efficientnet_b3": tv_models.efficientnet_b3,
            "efficientnet_b4": tv_models.efficientnet_b4,
            "efficientnet_b5": tv_models.efficientnet_b5,
            "efficientnet_b6": tv_models.efficientnet_b6,
            "efficientnet_b7": tv_models.efficientnet_b7,
        }
        if name not in builders:
            raise ValueError(f"TorchVisionEfficientNet: name doit être dans {list(builders.keys())}, reçu {name}")
        backbone = builders[name](weights=None)
        _replace_tv_classifier(backbone, num_classes, "classifier")
        self.model = backbone

    def forward(self, x):
        return self.model(x)


class TorchVisionConvNeXt(nn.Module):
    """
    ConvNeXt torchvision (Tiny, Small, Base, Large) pour entraînement / prédiction.
    state_dict : model.features.*, model.avgpool, model.classifier.*
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        name = (cfg.get("name") or "convnext_tiny").lower()
        builders = {
            "convnext_tiny": tv_models.convnext_tiny,
            "convnext_small": tv_models.convnext_small,
            "convnext_base": tv_models.convnext_base,
            "convnext_large": tv_models.convnext_large,
        }
        if name not in builders:
            raise ValueError(f"TorchVisionConvNeXt: name doit être dans {list(builders.keys())}, reçu {name}")
        backbone = builders[name](weights=None)
        _replace_tv_classifier(backbone, num_classes, "classifier")
        self.model = backbone

    def forward(self, x):
        return self.model(x)


class DropPath(nn.Module):
    """Stochastic depth: randomly drops entire residual branches during training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype).floor_().clamp_(0, 1)
        return x * mask / keep


class SqueezeExcitation(nn.Module):
    """Channel attention: learns per-channel importance weights."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """
    Residual block:  Conv-BN-Act → Conv-BN → SE → DropPath → + skip
    Optionally downsamples spatially (stride=2) and changes channel width.
    """

    def __init__(self, cin, cout, stride=1, drop_path=0.0, se_reduction=4):
        super().__init__()

        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)

        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

        self.act = nn.GELU()
        self.se = SqueezeExcitation(cout, reduction=se_reduction)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        skip = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out)

        return self.act(out + skip)


class ModernCNN(nn.Module):
    """
    A ResNet-style CNN built entirely from scratch with modern tricks:
      - Convolutional stem (no aggressive 7×7 / maxpool)
      - Residual blocks with skip connections
      - Squeeze-and-Excitation channel attention
      - Stochastic depth (DropPath) regularization
      - GELU activations
      - Global Average Pooling + dropout + classifier

    Config keys (all optional, sensible defaults provided):
        base_channels  : int   – width of first stage (default 64)
        depths         : list  – number of blocks per stage (default [2,2,3,3])
        drop_path      : float – max stochastic depth rate (default 0.15)
        dropout        : float – classifier dropout (default 0.3)
        se_reduction   : int   – SE squeeze ratio (default 4)
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()

        in_ch = input_size[0]
        base = cfg.get("base_channels", 64)
        depths = cfg.get("depths", [2, 2, 3, 3])
        max_drop = cfg.get("drop_path", 0.15)
        dropout = cfg.get("dropout", 0.3)
        se_red = cfg.get("se_reduction", 4)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base // 2),
            nn.GELU(),
            nn.Conv2d(base // 2, base, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
        )

        total_blocks = sum(depths)
        block_idx = 0
        stages = []
        ch_in = base

        for stage_id, n_blocks in enumerate(depths):
            ch_out = base * (2 ** stage_id)
            blocks = []
            for i in range(n_blocks):
                stride = 2 if (i == 0 and stage_id > 0) else 1
                dp_rate = max_drop * block_idx / max(total_blocks - 1, 1)
                blocks.append(
                    ResBlock(ch_in, ch_out, stride=stride,
                             drop_path=dp_rate, se_reduction=se_red)
                )
                ch_in = ch_out
                block_idx += 1
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(ch_in, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)
