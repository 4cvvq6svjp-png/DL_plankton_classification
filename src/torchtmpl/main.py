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

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # TODO : Define the loss function
    loss = None
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Build the optimizer
    logging.info("= Optimizer")
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # TODO : Define the optimizer
    optimizer = None
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train_one_epoch(model, train_loader, loss, optimizer, device)

        # Test
        test_loss = utils.test(model, valid_loader, loss, device)

        updated = model_checkpoint.update(test_loss)
        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {"train_CE": train_loss, "test_CE": test_loss}

        # On the tensorboard
        for key, value in metrics.items():
            tensorboard_writer.add_scalar(key, value, e)



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
