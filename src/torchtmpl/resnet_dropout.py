# coding: utf-8
"""
Standalone script that implements a ResNet with dropout and an
"ensemble bagging" training loop using majority voting on the
validation set.

This module reuses the same data loaders as :mod:`torchtmpl.main`
and therefore works with the same configuration file structure.

Usage example::

    python -m torchtmpl.resnet_dropout config.yaml

The config must contain the usual ``data`` and ``optim`` entries
and additionally an ``ensemble`` section::

    model:
      dropout: 0.4           # probability used in the final dropout layer
    ensemble:
      n_models: 5            # number of independently trained networks

The script will train ``n_models`` ResNet18 instances, each with a
separate optimizer and checkpoint.  After training all models the
validation set is evaluated by majority voting across the ensemble.

The training code is deliberately simple; it borrows the standard
``train_one_epoch`` and ``test`` helpers from :mod:`torchtmpl.utils`.
"""

import logging
import os
import sys
import datetime

import yaml
import torch
import torch.nn as nn
from torchvision.models import resnet18

from . import data, utils


# ---------------------------------------------------------------------------
# model definition
# ---------------------------------------------------------------------------

class ResNetDropout(nn.Module):
    """A small wrapper around ``torchvision``'s ResNet18 with a dropout
    layer inserted just before the final fully-connected classifier.

    The network takes a ``dropout`` probability argument.  All other
    defaults are identical to the original ResNet18 implementation.
    """

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.backbone = resnet18(pretrained=False)
        in_feat = self.backbone.fc.in_features
        # replace the last fully connected layer by a sequential block
        # implementing dropout -> linear.
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# helpers for ensemble evaluation
# ---------------------------------------------------------------------------

def majority_vote_accuracy(models, loader, device):
    """Compute accuracy on ``loader`` by majority voting across ``models``.

    Models are expected to be in ``eval()`` mode and already moved to
    the correct ``device``.  The function returns the average accuracy
    (0.0–1.0) obtained by taking the element-wise mode of the
    individual predictions.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # collect predictions from every ensemble member
            preds = []
            for m in models:
                outputs = m(inputs)
                preds.append(torch.argmax(outputs, dim=1).cpu())
            preds = torch.stack(preds, dim=0)  # shape (n_models, batch)

            voted, _ = torch.mode(preds, dim=0)
            voted = voted.to(device)

            correct += (voted == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# training loop for a single network (helper)
# ---------------------------------------------------------------------------


def train_single_model(model, train_loader, valid_loader, optimizer, loss_f,
                       epochs, device, logdir):
    """Train ``model`` for ``epochs`` using the provided loaders and
    optimizer.  A ``ModelCheckpoint`` is created so that the best weights
    (according to validation loss) are saved underneath ``logdir``.
    The function returns the model instance after training (it will have
    the best weights loaded).
    """
    checkpoint = utils.ModelCheckpoint(
        model, os.path.join(logdir, "best_model.pt"), min_is_best=True
    )

    for epoch in range(epochs):
        train_loss = utils.train_one_epoch(model, train_loader, loss_f,
                                          optimizer, device)
        val_loss = utils.test(model, valid_loader, loss_f, device)

        updated = checkpoint.update(val_loss)
        logging.info(
            f"[{epoch}/{epochs}] train={train_loss:.4f} "
            f"val={val_loss:.4f} "
            f"{'[BEST]' if updated else ''}"
        )

    # load best weights before returning
    model.load_state_dict(torch.load(checkpoint.savepath, map_location=device))
    return model


# ---------------------------------------------------------------------------
# ensemble training entry point
# ---------------------------------------------------------------------------


def train_ensemble(config):
    """Main entry point for training an ensemble of ResNetDropout models."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader, test_loader, input_size, num_classes = \
        data.get_dataloaders(config["data"], use_cuda)

    ensemble_cfg = config.get("ensemble", {})
    n_models = ensemble_cfg.get("n_models", 5)
    dropout = config.get("model", {}).get("dropout", 0.5)

    optim_cfg = config.get("optim", {})
    lr = optim_cfg.get("params", {}).get("lr", 1e-3)

    # create a root directory for the whole run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_logdir = utils.generate_unique_logpath(
        config["logging"]["logdir"], f"ResNetDropout_{timestamp}"
    )
    os.makedirs(root_logdir, exist_ok=True)

    # loss function used for training/validation
    loss_f = nn.CrossEntropyLoss()

    models_list = []

    for i in range(n_models):
        logging.info(f"Training model {i+1}/{n_models}")
        model = ResNetDropout(num_classes, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # each model gets its own subdirectory
        subdir = os.path.join(root_logdir, f"member_{i}")
        os.makedirs(subdir, exist_ok=True)

        model = train_single_model(
            model, train_loader, valid_loader, optimizer, loss_f,
            config["nepochs"], device, subdir
        )
        models_list.append(model)

    # once every network has been trained, compute ensemble performance
    logging.info("Evaluating ensemble on validation set (majority vote)")
    acc = majority_vote_accuracy(models_list, valid_loader, device)
    logging.info(f"Ensemble validation accuracy: {acc*100:.2f}%")

    # optionally compute ensemble loss by averaging logits
    # this part is just for information and not used for checkpointing
    ensemble_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits_sum = None
            for m in models_list:
                out = m(inputs)
                logits_sum = out if logits_sum is None else logits_sum + out

            # average logits before computing cross entropy
            avg_logits = logits_sum / len(models_list)
            ensemble_loss += loss_f(avg_logits, targets).item() * targets.size(0)
            n_samples += targets.size(0)
    ensemble_loss /= n_samples
    logging.info(f"Ensemble validation CE loss: {ensemble_loss:.4f}")


# ---------------------------------------------------------------------------
# command line interface
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print("Usage: python -m torchtmpl.resnet_dropout config.yaml")
        sys.exit(1)

    cfg = yaml.safe_load(open(sys.argv[1], "r"))
    train_ensemble(cfg)
