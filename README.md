# Deep Learning — Classification de plancton (ZooCam Challenge)

Pipeline PyTorch modulaire pour la classification d’images de plancton. Le projet vise la **modularité** (modèles, optimiseurs, pertes configurables) et la **reproductibilité** (configs sauvegardées, logs par run, environnements isolés).

---

## Vue d’ensemble du pipeline

```
Données (train/)  →  Transforms & DataLoaders  →  Modèles (CNN / backbones pré-entraînés)
                                                              ↓
                                                    Entraînement (warmup + fine-tuning)
                                                              ↓
                                                    Checkpoints + TensorBoard
                                                              ↓
                                                    Prédiction (single ou ensemble pondéré)
                                                              ↓
                                                    submission.csv
```

### Étapes principales

1. **Données**  
   Structure attendue : `root_dir/train/<classe>/<images>` et `root_dir/test/imgs/`.  
   Split train/validation stratifié, échantillonnage pondéré pour gérer le déséquilibre des classes.

2. **Transforms**  
   Préprocessing (resize, normalisation ImageNet) et augmentations en entraînement (RandomResizedCrop, flips, rotation, ColorJitter, RandomErasing, etc.).

3. **Modèles**  
   Modèles disponibles via la config : `HfModel` (Hugging Face, ex. ResNet-50), `TorchVisionResNet`, `PretrainedEfficientNet`, `PretrainedConvNeXtTiny`, ou `VanillaCNN`.

4. **Entraînement**  
   - **Phase 1 — Warm-up** (optionnel) : seule la tête est entraînée, backbone gelé.  
   - **Phase 2 — Fine-tuning** : tout le réseau avec LR distincts (backbone / tête), Focal Loss ou CrossEntropy, Mixup/CutMix optionnels, scheduler (CosineAnnealing ou ReduceLROnPlateau).  
   Métrique cible : **macro F1** sur la validation. Checkpoints et résumés sauvegardés dans un répertoire unique par run.

5. **Prédiction**  
   - **Modèle unique** : chargement d’un checkpoint et génération de `submission.csv`.  
   - **Ensemble pondéré** : plusieurs modèles avec poids dérivés des F1 de validation ; vote par softmax pondéré.

---

## Structure du projet

```
dl_plankton_team7/
├── config-resnet50.yaml      # Exemple ResNet-50 (Hugging Face)
├── config-effnet.yaml        # Exemple EfficientNet
├── config-convnext.yaml      # Exemple ConvNeXt
├── config-ensemble-weighted.yaml   # Config pour l’ensemble pondéré
├── src/torchtmpl/
│   ├── main.py               # Point d’entrée : train / test
│   ├── data.py               # DataLoaders, split, KaggleTestDataset
│   ├── transforms.py         # Pipelines train/valid/test
│   ├── optim.py              # Optimiseur, scheduler, Focal Loss
│   ├── utils.py              # train_one_epoch, test, ModelCheckpoint
│   ├── predict.py            # Prédiction modèle unique → submission.csv
│   ├── predict_ensemble.py   # Ensemble (prédictions multiples)
│   ├── predict_ensemble_weighted.py  # Ensemble pondéré par F1
│   └── models/
│       ├── __init__.py       # build_model()
│       ├── base_models.py
│       └── cnn_models.py     # HfModel, TorchVisionResNet, EfficientNet, ConvNeXt, VanillaCNN
├── submit-slurm.py           # Soumission SLURM (cluster)
├── run_test.sh               # Exemple de script de test
├── pyproject.toml
└── setup.py
```

---

## Installation

```bash
python3 -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -e .
```

Les dépendances (PyTorch, torchvision, transformers, scikit-learn, etc.) sont définies dans `pyproject.toml`.

---

## Configuration (YAML)

Chaque run est piloté par un fichier YAML. Exemple (extrait) :

```yaml
data:
  root_dir: 'path/to/dataset'
  batch_size: 32
  num_workers: 8
  valid_ratio: 0.2
  img_size: 224
  resize_size: 256

nepochs: 40
warmup_epochs: 5

model:
  class: HfModel
  name: microsoft/resnet-50
  freeze_backbone: False

loss:
  class: FocalLoss
  gamma: 1.0

optim:
  algo: AdamW
  head_lr: 0.001
  backbone_lr: 0.00001
  finetune_head_lr: 0.0001
  scheduler: CosineAnnealing

logging:
  logdir: "./logs"
```

- **data** : répertoire des données, taille d’image, batch, workers, ratio de validation.  
- **model** : `class` (HfModel, TorchVisionResNet, etc.) et `name` (ex. `microsoft/resnet-50`).  
- **loss** : `FocalLoss` (avec gamma, optionnellement poids de classes) ou CrossEntropy.  
- **optim** : LR séparés pour backbone/tête, scheduler, optionnellement Mixup/CutMix dans une section `mix`.

---

## Entraînement

```bash
python -m torchtmpl.main config-resnet50.yaml train
```

- Les logs et checkpoints sont écrits dans `logging.logdir` avec un sous-dossier unique (timestamp + nom du modèle).  
- Chaque run enregistre `config.yaml`, `summary.txt`, `best_model.pt` et les checkpoints intermédiaires.  
- TensorBoard : `tensorboard --logdir ./logs`.

Reprise depuis un checkpoint :

```yaml
resume:
  checkpoint: path/to/run/last_checkpoint.pt
```

---

## Prédiction

### Modèle unique

```bash
python -m torchtmpl.predict config-resnet50.yaml path/to/best_model.pt
```

Génère `submission.csv` (colonnes `imgname`, `label`) à partir du dossier `root_dir/test/imgs/`.

### Ensemble pondéré par F1

Le fichier `config-ensemble-weighted.yaml` liste plusieurs modèles avec leur config, checkpoint et score F1. Les poids sont normalisés à partir des F1.

```bash
python -m torchtmpl.predict_ensemble_weighted config-ensemble-weighted.yaml
# or with a custom output file:
python -m torchtmpl.predict_ensemble_weighted config-ensemble-weighted.yaml path/to/submission.csv
```

---

## Exécution sur cluster (SLURM)

Le script `submit-slurm.py` soumet l’entraînement sur un cluster SLURM (à adapter selon votre environnement) :

```bash
python submit-slurm.py config-resnet50.yaml [nruns]
```

- Vérifie que les modifications sont commitées.  
- Copie le code, crée un venv, lance `python -m torchtmpl.main <config> train` et récupère les logs.

---

## Tests des modules

Les modules exposent des petits tests via `__main__.py`. Exemple pour les modèles :

```bash
python -m torchtmpl.models
```

Pour tester les dataloaders avec un config donné :

```bash
python -m torchtmpl.data config-resnet50.yaml
```

(Un aperçu de batch peut être sauvegardé en image selon le code dans `data.py`.)

---

## Résumé des commandes

| Action              | Commande |
|---------------------|----------|
| Installer           | `pip install -e .` |
| Entraîner           | `python -m torchtmpl.main <config.yaml> train` |
| Prédire (1 modèle)  | `python -m torchtmpl.predict <config.yaml> <checkpoint.pt>` |
| Prédire (ensemble)  | `python -m torchtmpl.predict_ensemble_weighted <ensemble.yaml> [out.csv]` |
| Soumettre SLURM     | `python submit-slurm.py <config.yaml> [nruns]` |
| Tester les modèles  | `python -m torchtmpl.models` |

---

## Licence

Voir le fichier `LICENSE` du dépôt.
