# coding: utf-8

import logging
import os
import sys
import datetime
import yaml
import torch
import torch.nn as nn
from torchvision.models import resnet18


# ==========================================================
# MODEL
# ==========================================================

class ResNetDropout(nn.Module):
    """
    ResNet18 with a dropout layer before the final classifier.

    Compatible with build_model(cfg, input_size, num_classes)
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()

        # dropout récupéré depuis config["model"]["dropout"]
        dropout = cfg.get("dropout", 0.5)

        self.backbone = resnet18(weights=None)
        in_feat = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ==========================================================
# ENSEMBLE UTILITIES
# ==========================================================

def majority_vote_accuracy(models, loader, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = []
            for m in models:
                outputs = m(inputs)
                preds.append(torch.argmax(outputs, dim=1))

            preds = torch.stack(preds, dim=0)
            voted, _ = torch.mode(preds, dim=0)

            correct += (voted == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


# ==========================================================
# ENSEMBLE TRAINING
# ==========================================================

def train_ensemble(config):

    # 🔥 IMPORTS ICI pour éviter import circulaire
    from .. import data, utils

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader, test_loader, input_size, num_classes = \
        data.get_dataloaders(config["data"], use_cuda)

    ensemble_cfg = config.get("ensemble", {})
    n_models = ensemble_cfg.get("n_models", 5)

    optim_cfg = config.get("optim", {})
    lr = optim_cfg.get("params", {}).get("lr", 1e-3)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_logdir = utils.generate_unique_logpath(
        config["logging"]["logdir"],
        f"ResNetDropout_{timestamp}"
    )
    os.makedirs(root_logdir, exist_ok=True)

    loss_f = nn.CrossEntropyLoss()
    models_list = []

    for i in range(n_models):
        logging.info(f"Training model {i+1}/{n_models}")

        model = ResNetDropout(
            config["model"],
            input_size,
            num_classes
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        subdir = os.path.join(root_logdir, f"member_{i}")
        os.makedirs(subdir, exist_ok=True)

        checkpoint = utils.ModelCheckpoint(
            model,
            os.path.join(subdir, "best_model.pt"),
            min_is_best=True
        )

        for epoch in range(config["nepochs"]):

            train_loss = utils.train_one_epoch(
                model, train_loader, loss_f, optimizer, device
            )

            val_loss = utils.test(
                model, valid_loader, loss_f, device
            )

            updated = checkpoint.update(val_loss)

            logging.info(
                f"[{epoch}/{config['nepochs']}] "
                f"train={train_loss:.4f} "
                f"val={val_loss:.4f} "
                f"{'[BEST]' if updated else ''}"
            )

        model.load_state_dict(
            torch.load(checkpoint.savepath, map_location=device)
        )

        models_list.append(model)

    logging.info("Evaluating ensemble")
    acc = majority_vote_accuracy(models_list, valid_loader, device)
    logging.info(f"Ensemble validation accuracy: {acc*100:.2f}%")


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print("Usage: python -m torchtmpl.resnet_dropout config.yaml")
        sys.exit(1)

    cfg = yaml.safe_load(open(sys.argv[1], "r"))
    train_ensemble(cfg)