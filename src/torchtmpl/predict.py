# coding: utf-8
import os
import sys
import yaml
import logging
import torch
import pandas as pd
from torchvision.transforms import v2

from . import data
from . import models
from . import transforms as custom_transforms

def generate_submission(config, checkpoint_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    logging.info(f"Using device: {device}")

    data_config = config["data"]
    test_dir = os.path.join(data_config["root_dir"], "test", "imgs")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Le dossier test est introuvable : {test_dir}")

    img_size = data_config.get("img_size", 224)
    resize_size = data_config.get("resize_size", 256)
    preprocess_transforms = custom_transforms.get_transforms(
        split="test", img_size=img_size, resize_size=resize_size
    )

    test_dataset = data.KaggleTestDataset(test_dir, transform=preprocess_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=data_config["batch_size"], 
        shuffle=False, 
        num_workers=data_config["num_workers"]
    )
    
    logging.info(f"Loaded {len(test_dataset)} images for prediction.")

    num_classes = config.get("num_classes", 86)

    input_size = (3, img_size, img_size)
    model = models.build_model(config["model"], input_size, num_classes)
    
    logging.info(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_filenames = []
    all_predictions = []
    logging.info("Starting inference...")
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, 1)
            
            all_filenames.extend(filenames)
            all_predictions.extend(predicted_classes.cpu().numpy())

    df = pd.DataFrame({
        "imgname": all_filenames,
        "label": all_predictions
    })
    df = df.sort_values("imgname").reset_index(drop=True)

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