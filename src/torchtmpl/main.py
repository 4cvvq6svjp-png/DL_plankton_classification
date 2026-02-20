# coding: utf-8

import logging
import sys
import os
import pathlib
import csv
from tqdm import tqdm
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
import torchinfo.torchinfo as torchinfo

from . import data
from . import models
from . import utils


# ==========================================================
# TRAIN
# ==========================================================

def train(config):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    logging.info("= Building dataloaders")

    train_loader, valid_loader, test_loader, input_size, num_classes = \
        data.get_dataloaders(config["data"], use_cuda)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = models.build_model(config["model"], input_size, num_classes)
    model.to(device)

    # -----------------------------
    # LOSS
    # -----------------------------
    loss = torch.nn.CrossEntropyLoss()

    # -----------------------------
    # OPTIMIZER (FIX LR BUG)
    # -----------------------------
    optim_config = config.get("optim", {})
    lr = optim_config.get("params", {}).get("lr", 1e-3)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # LOGGING
    # -----------------------------
    logdir = config["logging"]["logdir"]

    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    model_checkpoint = utils.ModelCheckpoint(
        model,
        os.path.join(logdir, "best_model.pt"),
        min_is_best=True
    )

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for epoch in range(config["nepochs"]):

        train_loss = utils.train_one_epoch(
            model, train_loader, loss, optimizer, device
        )

        val_loss = utils.test(
            model, valid_loader, loss, device
        )

        updated = model_checkpoint.update(val_loss)

        logging.info(
            f"[{epoch}/{config['nepochs']}] "
            f"Train CE: {train_loss:.4f} "
            f"Val CE: {val_loss:.4f} "
            f"{'[BEST]' if updated else ''}"
        )

        writer.add_scalar("train_CE", train_loss, epoch)
        writer.add_scalar("val_CE", val_loss, epoch)


# ==========================================================
# TEST / INFERENCE
# ==========================================================

def test(config):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    _, _, test_loader, input_size, num_classes = \
        data.get_dataloaders(config["data"], use_cuda)

    model = models.build_model(
        config["model"],
        input_size,
        num_classes
    ).to(device)

    checkpoint_path = config["test"]["checkpoint"]
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    image_names = []
    predictions = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader):

            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            preds = preds.cpu().tolist()

            image_names.extend(filenames)
            predictions.extend(preds)

    submission_path = config["output"]["submission_path"]

    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imgname", "label"])

        for name, pred in zip(image_names, predictions):
            writer.writerow([name, int(pred)])

    logging.info(f"Submission saved to {submission_path}")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 3:
        print("Usage: python main.py config.yaml <train|test>")
        sys.exit(1)

    config = yaml.safe_load(open(sys.argv[1], "r"))
    command = sys.argv[2]

    if command == "train":
        train(config)
    elif command == "test":
        test(config)
    else:
        raise ValueError("Command must be train or test")