# coding: utf-8
import os
import sys
import yaml
import logging
import torch
import pandas as pd
from torchvision.transforms import v2

# Local imports
from . import data
from . import models
from . import transforms as custom_transforms

def generate_submission(config, checkpoint_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")

    # 1. Préparation des données
    data_config = config["data"]
    test_dir = os.path.join(data_config["root_dir"], "test", "imgs")
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Le dossier test est introuvable : {test_dir}")

    # Les mêmes transformations qu'à l'entraînement (sans l'augmentation)
    preprocess_transforms = custom_transforms.get_transforms(split="test", img_size=224)

    test_dataset = data.KaggleTestDataset(test_dir, transform=preprocess_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=data_config["batch_size"], 
        shuffle=False, 
        num_workers=data_config["num_workers"]
    )
    
    logging.info(f"Loaded {len(test_dataset)} images for prediction.")

    # 2. Chargement du modèle
    # Attention: Il faut donner le bon nombre de classes ! 
    # Remplace 86 par le vrai nombre de classes de ton dataset d'entraînement.
    num_classes = 86 
    
    model = models.build_model(config["model"], (3, 224, 224), num_classes)
    
    logging.info(f"Loading weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval() # Mode évaluation

    # 3. Boucle de prédiction
    all_filenames = []
    all_predictions = []
    logging.info("Starting inference...")
    with torch.no_grad(): # Indispensable pour ne pas faire de OutOfMemory !
        for i, (images, filenames) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            
            # DEBUG : Afficher les sorties du modèle
            logging.info(f"Batch {i}: outputs shape = {outputs.shape}")
            logging.info(f"Batch {i}: outputs min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
            logging.info(f"Batch {i}: Sample output = {outputs[0]}")
            
            # On prend l'indice de la classe ayant la plus forte probabilité (les logits les plus hauts)
            _, predicted_classes = torch.max(outputs, 1)
            logging.info(f"Batch {i}: Predicted classes = {predicted_classes[:5]}")
            
            all_filenames.extend(filenames)
            all_predictions.extend(predicted_classes.cpu().numpy())
            
            if i >= 2:  # Afficher seulement les 3 premiers batches
                break
    # 4. Sauvegarde du fichier CSV
    # Vérifie sur Kaggle le nom exact des colonnes attendues (souvent "Id" ou "image_id" et "Category" ou "label")
    df = pd.DataFrame({
        "image_id": all_filenames,
        "label": all_predictions
    })
    
    output_filename = "submission.csv"
    df.to_csv(output_filename, index=False)
    logging.info(f"Success! Predictions saved to {output_filename}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : python -m torchtmpl.predict config.yaml path/to/best_model.pt")
        sys.exit(-1)

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    
    logging.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    generate_submission(config, checkpoint_path)