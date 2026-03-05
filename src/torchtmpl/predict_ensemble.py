#! /usr/bin/env python
# coding: utf-8

import os
import sys
import logging
import yaml

import torch
import pandas as pd

from . import data
from . import models
from . import transforms as custom_transforms


def _build_test_loader_from_config(config, device, batch_size_override=None):
    """Construit un DataLoader test à partir d'une config YAML.
    Si batch_size_override est donné, il est utilisé à la place de data.batch_size
    (nécessaire en ensemble pour que tous les loaders aient les mêmes batches).
    """
    data_config = config["data"]
    test_dir = os.path.join(data_config["root_dir"], "test", "imgs")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Le dossier test est introuvable : {test_dir}")

    img_size = data_config.get("img_size", 224)
    resize_size = data_config.get("resize_size", 256)
    preprocess_transforms = custom_transforms.get_transforms(
        split="test", img_size=img_size, resize_size=resize_size
    )

    batch_size = batch_size_override if batch_size_override is not None else data_config["batch_size"]
    test_dataset = data.KaggleTestDataset(test_dir, transform=preprocess_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config["num_workers"],
    )

    logging.info(
        f"[{config['model']['name']}] Loaded {len(test_dataset)} images for prediction "
        f"(img_size={img_size}, resize_size={resize_size})"
    )

    num_classes = config.get("num_classes", 86)
    input_size = (3, img_size, img_size)

    return test_loader, num_classes, input_size


def generate_ensemble_submission(model_specs, output_filename="submission_ensemble.csv"):
    """
    Génère une submission Kaggle à partir d'un ensemble hétérogène :
    chaque modèle peut avoir sa propre config (EffNet, ConvNeXt, ResNet, ...).

    model_specs: liste de dicts
        [{"config": cfg1, "ckpt": path1}, {"config": cfg2, "ckpt": path2}, ...]
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")

    if not model_specs:
        raise ValueError("Aucun modèle fourni pour l'ensemble.")

    # Tous les loaders doivent utiliser le même batch_size pour que les batches soient alignés
    common_batch_size = model_specs[0]["config"]["data"].get("batch_size", 32)

    # 1. Pour chaque modèle : DataLoader + modèle
    loaders = []
    models_list = []

    for spec in model_specs:
        cfg = spec["config"]
        ckpt_path = spec["ckpt"]

        test_loader, num_classes, input_size = _build_test_loader_from_config(
            cfg, device, batch_size_override=common_batch_size
        )
        loaders.append(test_loader)

        logging.info(f"Loading model '{cfg['model']['name']}' from {ckpt_path}")
        model = models.build_model(cfg["model"], input_size, num_classes)

        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models_list.append(model)

    # Vérification très simple : même nombre de batches et même ordre de fichiers
    # grâce à KaggleTestDataset (tri des noms et shuffle=False).
    all_filenames = []
    all_predictions = []
    logging.info("Starting heterogeneous ensemble inference...")

    with torch.no_grad():
        # zip sur les loaders : à chaque itération, on récupère un batch par modèle
        for batches in zip(*loaders):
            # batches[i] = (images_i, filenames_i) pour le modèle i
            images_0, filenames_0 = batches[0]
            images_0 = images_0.to(device)

            # On accumule les probabilités dans l'espace du premier modèle
            probs_sum = None

            for (model, (images_i, filenames_i)) in zip(models_list, batches):
                # par sécurité, on vérifie que les noms d'images sont alignés
                if filenames_i != filenames_0:
                    raise RuntimeError("Désalignement des batchs entre les DataLoaders d'ensemble.")

                images_i = images_i.to(device)
                outputs = model(images_i)
                probs = torch.softmax(outputs, dim=1)

                if probs_sum is None:
                    probs_sum = probs
                else:
                    probs_sum = probs_sum + probs

            _, predicted_classes = torch.max(probs_sum, 1)

            all_filenames.extend(filenames_0)
            all_predictions.extend(predicted_classes.cpu().numpy())

    # 2. Sauvegarde CSV
    df = pd.DataFrame({"imgname": all_filenames, "label": all_predictions})
    df = df.sort_values("imgname").reset_index(drop=True)

    df.to_csv(output_filename, index=False)
    logging.info(f"Success! Ensemble predictions saved to {output_filename}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    # On attend des paires (config, checkpoint) : cfg1 ckpt1 cfg2 ckpt2 ...
    if (len(sys.argv) - 1) < 2 or (len(sys.argv) - 1) % 2 != 0:
        logging.error(
            "Usage : python -m torchtmpl.predict_ensemble "
            "config_effnet.yaml effnet_best.pt "
            "config_convnext.yaml convnext_best.pt "
            "[config_resnet.yaml resnet_best.pt ...]"
        )
        sys.exit(-1)

    args = sys.argv[1:]
    model_specs = []

    for i in range(0, len(args), 2):
        cfg_path = args[i]
        ckpt_path = args[i + 1]

        logging.info(f"Loading config from {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        model_specs.append({"config": cfg, "ckpt": ckpt_path})

    output_name = f"submission_ensemble_{len(model_specs)}models.csv"
    generate_ensemble_submission(model_specs, output_filename=output_name)
