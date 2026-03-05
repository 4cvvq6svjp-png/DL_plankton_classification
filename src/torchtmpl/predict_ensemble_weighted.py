#! /usr/bin/env python
# coding: utf-8

"""
Ensemble pondéré par F1 : chaque modèle contribue aux prédictions
selon son score F1 (configurable dans un fichier YAML).
"""

import os
import sys
import logging
import yaml

import torch
import pandas as pd

from . import models
from .predict_ensemble import _build_test_loader_from_config


def generate_weighted_ensemble_submission(
    model_specs,
    output_filename="submission_ensemble_weighted.csv",
):
    """
    Génère une submission par voting pondéré (soft) : pour chaque image,
    probs = sum(weight_i * softmax(logits_i)), puis argmax(probs).
    Les poids sont dérivés des F1 (normalisés pour sommer à 1).

    model_specs: liste de dicts avec clés:
        - config: dict de config (ou path vers YAML)
        - checkpoint: path vers .pt
        - f1_score: float (ex. 0.65), utilisé comme poids
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")

    if not model_specs:
        raise ValueError("Aucun modèle fourni pour l'ensemble.")

    # Poids à partir des F1 (normalisés)
    f1_scores = [float(spec["f1_score"]) for spec in model_specs]
    total = sum(f1_scores)
    weights = [f1 / total for f1 in f1_scores]
    logging.info(f"Weights (from F1, normalized): {[f'{w:.4f}' for w in weights]}")

    common_batch_size = model_specs[0]["config"]["data"].get("batch_size", 32)
    loaders = []
    models_list = []

    for spec, weight in zip(model_specs, weights):
        cfg = spec["config"]
        ckpt_path = spec["checkpoint"]

        test_loader, num_classes, input_size = _build_test_loader_from_config(
            cfg, device, batch_size_override=common_batch_size
        )
        loaders.append(test_loader)

        logging.info(
            f"Loading model '{cfg['model']['name']}' from {ckpt_path} (weight={weight:.4f})"
        )
        model = models.build_model(cfg["model"], input_size, num_classes)

        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models_list.append((model, weight))

    all_filenames = []
    all_predictions = []
    logging.info("Starting weighted ensemble inference...")

    with torch.no_grad():
        for batches in zip(*loaders):
            images_0, filenames_0 = batches[0]
            images_0 = images_0.to(device)
            probs_weighted = None

            for ((model, weight), (images_i, filenames_i)) in zip(models_list, batches):
                if filenames_i != filenames_0:
                    raise RuntimeError(
                        "Désalignement des batchs entre les DataLoaders d'ensemble."
                    )

                images_i = images_i.to(device)
                outputs = model(images_i)
                probs = torch.softmax(outputs, dim=1)

                if probs_weighted is None:
                    probs_weighted = weight * probs
                else:
                    probs_weighted = probs_weighted + weight * probs

            _, predicted_classes = torch.max(probs_weighted, 1)

            all_filenames.extend(filenames_0)
            all_predictions.extend(predicted_classes.cpu().numpy())

    df = pd.DataFrame({"imgname": all_filenames, "label": all_predictions})
    df = df.sort_values("imgname").reset_index(drop=True)

    df.to_csv(output_filename, index=False)
    logging.info(f"Success! Weighted ensemble predictions saved to {output_filename}")


def load_specs_from_yaml(yaml_path):
    """
    Charge le fichier YAML d'ensemble pondéré et retourne une liste de specs.
    Format attendu:
        models:
          - config: path/to/config.yaml
            checkpoint: path/to/best_model.pt
            f1_score: 0.65
          - ...
    Chaque config est chargée depuis le fichier et ajoutée au spec.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    models_list = data.get("models", data)
    if not models_list:
        raise ValueError(f"Aucune entrée 'models' dans {yaml_path}")

    specs = []
    for i, entry in enumerate(models_list):
        config_path = entry["config"]
        checkpoint_path = entry["checkpoint"]
        f1_score = float(entry["f1_score"])

        logging.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        specs.append({
            "config": cfg,
            "checkpoint": checkpoint_path,
            "f1_score": f1_score,
        })

    return specs


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        logging.error(
            "Usage: python -m src.torchtmpl.predict_ensemble_weighted <ensemble.yaml> [output.csv]"
        )
        sys.exit(-1)

    yaml_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else "submission_ensemble_weighted.csv"

    if not os.path.isfile(yaml_path):
        logging.error(f"Fichier introuvable: {yaml_path}")
        sys.exit(-1)

    model_specs = load_specs_from_yaml(yaml_path)
    generate_weighted_ensemble_submission(model_specs, output_filename=output_name)
