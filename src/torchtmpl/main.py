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

    # Build the loss
    logging.info("= Loss")
    loss_cfg = config.get("loss", {"class": "CrossEntropyLoss"})
    loss_class_name = loss_cfg["class"]
    
    # Move class weights to device
    class_weights = class_weights.to(device)
    
    if loss_class_name == "FocalLoss":
        loss = optim.FocalLoss(
            alpha=class_weights,  # Use computed class weights
            gamma=loss_cfg.get("gamma", 2)
        )
    else:
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)

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

    # Define the early stopping callback (based on validation loss)
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    # ==========================================
    # PHASE 1 : WARM-UP (Entraînement de la tête)
    # ==========================================
    logging.info("\n=== DÉBUT PHASE 1 : WARM-UP (Backbone gelé) ===")
    warmup_epochs = 3 
    
    # 1. On s'assure que le backbone est gelé
    for param in model.model.base_model.parameters():
        param.requires_grad = False
        
    # 2. On crée un optimiseur UNIQUEMENT pour les paramètres dégelés (la tête)
    optimizer_warmup = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3, 
        weight_decay=1e-2
    )

    for e in range(warmup_epochs):
        train_metrics = utils.train_one_epoch(model, train_loader, loss, optimizer_warmup, device)
        test_metrics = utils.test(model, valid_loader, loss, device)
        
        updated = model_checkpoint.update(test_metrics['loss'])
        logging.info(
            f"[Warm-up {e+1}/{warmup_epochs}] "
            f"Train Loss: {train_metrics['loss']:.3f} | Train Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {test_metrics['loss']:.3f} | Val F1: {test_metrics['f1']:.3f} "
            f"{'[>> BETTER <<]' if updated else ''}"
        )
        
        # Enregistrement TensorBoard
        tensorboard_writer.add_scalar('Loss/Train', train_metrics['loss'], e)
        tensorboard_writer.add_scalar('Loss/Validation', test_metrics['loss'], e)
        tensorboard_writer.add_scalar('F1/Validation', test_metrics['f1'], e)
        tensorboard_writer.add_scalar('Learning_Rate', 1e-3, e)


    # ==========================================
    # PHASE 2 : FINE-TUNING (Entraînement complet)
    # ==========================================
    logging.info("\n=== DÉBUT PHASE 2 : FINE-TUNING (Backbone dégelé) ===")
    
    # On lit la config pour le scheduler
    optim_config = config.get("optim", {})
    use_scheduler = optim_config.get("scheduler") == "CosineAnnealing"
    
    # 1. On dégèle tout le modèle (Backbone + Tête)
    for param in model.parameters():
        param.requires_grad = True

    # 2. On recrée un NOUVEL optimiseur avec un LR beaucoup plus faible
    fine_tune_lr = 1e-5
    optimizer_finetune = torch.optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=1e-2)

    # 3. On attache le CosineAnnealingLR à ce nouvel optimiseur si demandé
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_finetune, 
            T_max=config["nepochs"] - warmup_epochs
        )

    for e in range(warmup_epochs, config["nepochs"]):
        train_metrics = utils.train_one_epoch(model, train_loader, loss, optimizer_finetune, device)
        
        if use_scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = fine_tune_lr

        test_metrics = utils.test(model, valid_loader, loss, device)
        
        updated = model_checkpoint.update(test_metrics['loss'])
        logging.info(
            f"[Fine-Tune {e+1}/{config['nepochs']}] LR: {current_lr:.6f} | "
            f"Train Loss: {train_metrics['loss']:.3f} | Train Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {test_metrics['loss']:.3f} | Val F1: {test_metrics['f1']:.3f} "
            f"{'[>> BETTER <<]' if updated else ''}"
        )
        
        # Enregistrement TensorBoard (suite continue de la courbe)
        tensorboard_writer.add_scalar('Loss/Train', train_metrics['loss'], e)
        tensorboard_writer.add_scalar('Loss/Validation', test_metrics['loss'], e)
        tensorboard_writer.add_scalar('F1/Validation', test_metrics['f1'], e)
        tensorboard_writer.add_scalar('Learning_Rate', current_lr, e)

    # Fermeture propre du logger TensorBoard
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
