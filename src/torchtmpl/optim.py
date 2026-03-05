# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with per-class weights (alpha) and label smoothing.
    gamma > 1 down-weights easy examples, focusing training on hard ones.
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none',
            weight=self.alpha, label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_loss(lossname):
    return eval(f"nn.{lossname}()")


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim


def build_scheduler(config, optimizer, finetune_epochs):
    """
    Fabrique de scheduler de LR pour la phase de fine-tuning.

    - supporte CosineAnnealingLR (par défaut, compatible configs historiques)
    - supporte ReduceLROnPlateau (scheduler: ReduceLROnPlateau)
    - gère finetune_warmup (nombre d'epochs de warmup linéaire)

    Retourne :
      - scheduler          : instance du scheduler
      - plateau_scheduler  : bool, True si ReduceLROnPlateau
      - finetune_warmup    : int, nb d'epochs de warmup pour le fine-tuning
      - initial_lrs        : liste des LR initiaux par param_group
    """
    optim_config = config.get("optim", {})

    scheduler_name = optim_config.get("scheduler", "CosineAnnealing")
    plateau_scheduler = scheduler_name == "ReduceLROnPlateau"
    finetune_warmup = config.get("finetune_warmup", 0)
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    if plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(optim_config.get("scheduler_factor", 0.5)),
            patience=int(optim_config.get("scheduler_patience", 5)),
            min_lr=float(optim_config.get("eta_min", 1e-6)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, finetune_epochs - finetune_warmup),
            eta_min=float(optim_config.get("eta_min", 1e-7)),
        )

    return scheduler, plateau_scheduler, finetune_warmup, initial_lrs
