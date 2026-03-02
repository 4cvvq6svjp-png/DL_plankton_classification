# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
from torch.utils.tensorboard import SummaryWriter
import datetime


# Local imports
from . import data
from . import models
from . import optim
from . import utils


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_size, num_classes, class_weights = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss (pas de class weights ici, le WeightedRandomSampler gère l'équilibrage)
    logging.info("= Loss")
    loss_cfg = config.get("loss", {"class": "CrossEntropyLoss"})
    loss_class_name = loss_cfg["class"]

    if loss_class_name == "FocalLoss":
        loss = optim.FocalLoss(
            alpha=None,
            gamma=loss_cfg.get("gamma", 1.0)
        )
    else:
        loss = torch.nn.CrossEntropyLoss()

    # Build the callbacks & logging
    logging_config = config["logging"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logname = f"{model_config['class']}_{timestamp}"
    
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    tensorboard_writer = SummaryWriter(logdir)

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset}\n"
        + f"Validation : {valid_loader.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

    # Sauvegarde du meilleur modèle basé sur le macro F1 (plus c'est haut, mieux c'est)
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    # ==========================================
    # PHASE 1 : WARM-UP (Entraînement de la tête uniquement)
    # ==========================================
    warmup_epochs = config.get("warmup_epochs", 5)
    head_lr = config.get("optim", {}).get("head_lr", 1e-3)
    logging.info(f"\n=== PHASE 1 : WARM-UP ({warmup_epochs} epochs, head LR={head_lr}) ===")

    for param in model.model.base_model.parameters():
        param.requires_grad = False

    optimizer_warmup = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=head_lr,
        weight_decay=1e-2,
    )

    for e in range(warmup_epochs):
        train_metrics = utils.train_one_epoch(model, train_loader, loss, optimizer_warmup, device)
        test_metrics = utils.test(model, valid_loader, loss, device)

        updated = model_checkpoint.update(test_metrics['f1'])
        logging.info(
            f"[Warm-up {e+1}/{warmup_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {test_metrics['loss']:.4f} | Val Macro-F1: {test_metrics['f1']:.4f} "
            f"{'[BEST]' if updated else ''}"
        )

        tensorboard_writer.add_scalar('Loss/Train', train_metrics['loss'], e)
        tensorboard_writer.add_scalar('Loss/Validation', test_metrics['loss'], e)
        tensorboard_writer.add_scalar('MacroF1/Validation', test_metrics['f1'], e)
        tensorboard_writer.add_scalar('Learning_Rate', head_lr, e)

    # ==========================================
    # PHASE 2 : FINE-TUNING avec LR différentiels
    # ==========================================
    optim_config = config.get("optim", {})
    backbone_lr = optim_config.get("backbone_lr", 1e-5)
    finetune_head_lr = optim_config.get("finetune_head_lr", 1e-4)
    finetune_epochs = config["nepochs"] - warmup_epochs

    logging.info(
        f"\n=== PHASE 2 : FINE-TUNING ({finetune_epochs} epochs, "
        f"backbone LR={backbone_lr}, head LR={finetune_head_lr}) ==="
    )

    for param in model.parameters():
        param.requires_grad = True

    backbone_params = list(model.model.base_model.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer_finetune = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": finetune_head_lr},
    ], weight_decay=1e-2)

    use_scheduler = optim_config.get("scheduler") == "CosineAnnealing"
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_finetune,
            T_0=max(finetune_epochs // 3, 5),
            T_mult=2,
            eta_min=1e-7,
        )

    for e in range(warmup_epochs, config["nepochs"]):
        train_metrics = utils.train_one_epoch(model, train_loader, loss, optimizer_finetune, device)

        if use_scheduler:
            scheduler.step(e - warmup_epochs)
            current_lr_bb = scheduler.get_last_lr()[0]
            current_lr_head = scheduler.get_last_lr()[1]
        else:
            current_lr_bb = backbone_lr
            current_lr_head = finetune_head_lr

        test_metrics = utils.test(model, valid_loader, loss, device)

        updated = model_checkpoint.update(test_metrics['f1'])
        logging.info(
            f"[Fine-Tune {e+1}/{config['nepochs']}] "
            f"BB-LR: {current_lr_bb:.2e} | Head-LR: {current_lr_head:.2e} | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {test_metrics['loss']:.4f} | Val Macro-F1: {test_metrics['f1']:.4f} "
            f"{'[BEST]' if updated else ''}"
        )

        tensorboard_writer.add_scalar('Loss/Train', train_metrics['loss'], e)
        tensorboard_writer.add_scalar('Loss/Validation', test_metrics['loss'], e)
        tensorboard_writer.add_scalar('MacroF1/Validation', test_metrics['f1'], e)
        tensorboard_writer.add_scalar('LR/Backbone', current_lr_bb, e)
        tensorboard_writer.add_scalar('LR/Head', current_lr_head, e)

    tensorboard_writer.close()
    logging.info("=== Entraînement terminé ! ===")



def test(config):
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
