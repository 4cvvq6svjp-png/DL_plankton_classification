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

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config.get("optim", {})
    lr = optim_config.get("lr", 1e-3)
    algo = optim_config.get("algo", "Adam")
    
    if algo == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    elif algo == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Build the Scheduler
    use_scheduler = optim_config.get("scheduler") == "CosineAnnealing"
    if use_scheduler:
        logging.info("= Using CosineAnnealingLR Scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["nepochs"])

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
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

    for e in range(config["nepochs"]):
            # Train 1 epoch
            train_metrics = utils.train_one_epoch(model, train_loader, loss, optimizer, device)
            
            # On met à jour le scheduler si on l'a activé
            if use_scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = lr

            # Test
            test_metrics = utils.test(model, valid_loader, loss, device)

            # Save best model based on validation loss
            updated = model_checkpoint.update(test_metrics['loss'])
            logging.info(
                "[%d/%d] LR: %.6f | "
                "Train Loss: %.3f | Train Acc: %.3f | Train F1: %.3f | "
                "Val Loss: %.3f | Val Acc: %.3f | Val F1: %.3f %s"
                % (
                    e,
                    config["nepochs"],
                    current_lr,
                    train_metrics['loss'],
                    train_metrics['accuracy'],
                    train_metrics['f1'],
                    test_metrics['loss'],
                    test_metrics['accuracy'],
                    test_metrics['f1'],
                    "[>> BETTER <<]" if updated else "",
                )
            )



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
