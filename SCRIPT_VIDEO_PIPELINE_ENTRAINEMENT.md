# Script vidéo — Pipeline d'entraînement (≈ 20 min)

*Script exhaustif pour expliquer chaque choix de la pipeline d'entraînement du projet dl_plankton_team7 (ZooCam Challenge).*

---

## 1. Introduction et contexte (≈ 1 min 30)

Bonjour. Dans cette vidéo je vais vous présenter l’intégralité de la pipeline d’entraînement que j’ai mise en place pour le projet de classification de plancton, dans le cadre du ZooCam Challenge. L’objectif est de classifier des images de plancton en un certain nombre de classes — dans notre cas 86 classes, déduites automatiquement des sous-dossiers du jeu d’entraînement. Je vais détailler chaque choix : données, modèles, fonction de perte, optimiseur, stratégie d’entraînement, et inférence, pour que vous compreniez pourquoi les choses sont faites ainsi et comment tout s’enchaîne.

---

## 2. Structure du projet (≈ 2 min)

Le code est organisé autour d’un package Python nommé **torchtmpl**, dans le dossier **src/torchtmpl**. C’est un template PyTorch modulaire pensé pour la reproductibilité : chaque run d’entraînement est enregistré dans un sous-dossier unique sous **logs**, avec la config utilisée, le résumé du modèle, les checkpoints et les courbes TensorBoard.

**Point d’entrée principal** : on lance l’entraînement avec  
`python -m torchtmpl.main <fichier_config.yaml> train`.  
Tout est piloté par un fichier YAML : pas besoin de modifier le code pour changer de modèle, de loss ou d’hyperparamètres.

Les **configs** peuvent être à la racine — par exemple `config-resnet50.yaml`, `config-convnext-torchvision.yaml` — ou copiées dans **configs/** lorsqu’on soumet un job sur le cluster SLURM via **submit-slurm.py**. Les runs produisent notamment : **config.yaml** (copie de la config), **summary.txt** (commande, config, torchinfo du modèle, infos datasets), **best_model.pt** (meilleur état du modèle selon le F1 validation), **last_checkpoint.pt** (dernier état pour reprise), et les logs TensorBoard.

---

## 3. Pipeline de données (≈ 4 min 30)

### 3.1 Chargement et structure

Les données d’entraînement sont chargées avec **torchvision.datasets.ImageFolder** : le répertoire **train** sous **root_dir** doit contenir un sous-dossier par classe, chaque image étant dans le dossier correspondant à sa classe. Le nombre de classes est donc **déduit automatiquement** du nombre de sous-dossiers — pour ZooCam, on obtient 86 classes. Aucun fichier de labels séparé n’est nécessaire.

Pour l’inférence, le jeu de test est géré par un dataset dédié **KaggleTestDataset** : il parcourt un répertoire d’images — typiquement **root_dir/test/imgs** —, trie les noms de fichiers pour la reproductibilité, et renvoie le tenseur image plus le nom du fichier, afin de produire un fichier de soumission avec les bons identifiants.

### 3.2 Découpage train / validation

On ne fait **pas** de split fixe par dossier : tout le contenu de **train** est chargé, puis on applique un **split stratifié** avec **sklearn.train_test_split**, avec un **valid_ratio** configurable — souvent 0,2, soit 20 % en validation. Le paramètre **stratify** garantit que les proportions de chaque classe sont respectées dans le train et la validation. Le **random_state** est fixé à 42 pour la reproductibilité. On obtient ainsi des indices train et validation ; le même **ImageFolder** est ensuite enveloppé dans des **Subset** pour ne charger que les échantillons correspondants.

### 3.3 Rééquilibrage des classes (WeightedRandomSampler)

Les classes de plancton sont souvent **déséquilibrées** : certaines sont très fréquentes, d’autres rares. Pour éviter que le modèle soit dominé par les classes majoritaires, on utilise un **WeightedRandomSampler** sur l’ensemble d’entraînement. Les poids d’échantillonnage sont de la forme **1 / (count_class^sampler_power)**. Le paramètre **sampler_power** est dans la config : à **0,5** on utilise la racine carrée des effectifs — un rééquilibrage doux qui favorise les classes rares sans forcer une distribution uniforme ; à **1,0** ce serait un rééquilibrage très agressif ; à **0** on désactive le rééquilibrage. On tire **num_samples = len(train)** avec **replacement=True** pour que chaque epoch voie bien tous les indices, mais avec des classes rares plus souvent représentées.

### 3.4 Transformations et augmentation (train vs valid/test)

Les transformations sont définies dans **transforms.py**, avec **get_transforms(split, img_size, resize_size)**.

**Base commune** : conversion en tenseur (**ToImage**), puis **Resize(resize_size)** — par exemple 256 — pour avoir une résolution intermédiaire avant le crop.

**En entraînement** :  
- **RandomResizedCrop** à **img_size** (souvent 224), avec scale (0,6, 1,0) et ratio (0,75, 1,33), pour varier la mise à l’échelle et le cadrage.  
- **RandomHorizontalFlip** et **RandomVerticalFlip** à probabilité 0,5 — pertinent pour du plancton qui n’a pas d’orientation canonique.  
- **RandomRotation** de 90 degrés.  
- **ColorJitter** sur luminosité, contraste et saturation (0,3 ; 0,3 ; 0,1) pour une certaine invariance aux conditions d’éclairage.  
- **RandomAffine** avec translation (0,1, 0,1) et scale (0,9, 1,1).  
- **GaussianBlur** (noyau 3, sigma entre 0,1 et 1,0) pour un peu de flou.  
- Puis **Normalize** avec les moyennes et écarts-types **ImageNet** (0,485, 0,456, 0,406 et 0,229, 0,224, 0,225) — cohérent avec les backbones pré-entraînés quand on en utilise.  
- **RandomErasing** avec probabilité 0,2 et scale (0,02, 0,15) pour de la régularisation supplémentaire.

**En validation et test** : pas d’augmentation. On fait **Resize** puis **CenterCrop(img_size)**, puis la même **Normalize** ImageNet. Les prédictions sont donc faites sur des images centrées, de taille fixe, reproductibles.

### 3.5 Poids de classes pour la loss

En plus du sampler, on calcule des **poids de classes** pour la fonction de perte : **compute_class_weights** utilise la formule **1 / sqrt(count + 1e-6)**, puis normalise pour que la somme des poids soit égale au nombre de classes. Ces poids sont utilisés par la **FocalLoss** (paramètre alpha) quand **use_class_weights** est activé, pour donner plus de poids aux classes rares dans le calcul de la perte.

---

## 4. Modèles (≈ 3 min 30)

### 4.1 Construction générique

Le modèle est construit par **models.build_model(config["model"], input_size, num_classes)**. La config contient une clé **class** qui est le nom de la classe du modèle — par exemple **TorchVisionResNet**, **TorchVisionConvNeXt**, **TorchVisionEfficientNet**, **HfModel**, etc. — et éventuellement **name** pour préciser la variante (resnet50, convnext_tiny, efficientnet_b0, etc.). Le code fait un **eval** sur cette classe après l’avoir importée depuis **base_models** et **cnn_models**, ce qui permet d’ajouter de nouveaux modèles sans toucher au main.

### 4.2 Choix d’architectures

On dispose de modèles “jouet” (**Linear**, **FFN**) et de vrais modèles pour la classification d’images :

- **TorchVisionResNet** : ResNet50 torchvision, sans poids ImageNet par défaut — **weights=None** —, avec la couche **fc** remplacée par une **Linear** vers **num_classes**. Les clés du state_dict sont de la forme **model.conv1**, **model.layer1** à **layer4**, **model.fc**, ce qui permet de charger des checkpoints entraînés avec cette API.

- **TorchVisionEfficientNet** : EfficientNet B0 à B7 ; la tête **classifier** est remplacée pour sortir **num_classes**.

- **TorchVisionConvNeXt** : ConvNeXt Tiny, Small, Base ou Large ; même principe, remplacement de la tête **classifier**.

- **HfModel** : modèles Hugging Face via **AutoModelForImageClassification.from_pretrained** — par exemple **google/efficientnet-b2** —, avec **num_labels=num_classes** et **ignore_mismatched_sizes=True**. On peut geler le backbone avec **freeze_backbone**.

- **ModernCNN** : un CNN “from scratch” avec des blocs type ResBlock, Squeeze-and-Excitation, DropPath, GELU — utile si on veut une architecture légère sans dépendance à un backbone pré-entraîné.

Le **nombre de classes** est toujours déterminé à l’entraînement par les dossiers de **train**. En inférence, si le dataset test n’a pas d’étiquettes, **num_classes** peut être fixé dans la config — par exemple 86 pour ZooCam.

### 4.3 Transfer learning et backbone / tête

Pour les modèles type ResNet, EfficientNet, ConvNeXt ou HfModel, on distingue **backbone** (extracteur de features) et **tête** (dernière couche vers les classes). Cette séparation est utilisée pour :  
- **deux learning rates distincts** en fine-tuning : backbone plus petit (ex. 1e-5 ou 3e-5), tête plus grand (ex. 1e-4 ou 3e-4) ;  
- une **phase de warm-up** optionnelle où seul le head est entraîné avec le backbone gelé.

---

## 5. Fonction de perte (≈ 2 min)

Deux options dans la config sous **loss.class**.

**CrossEntropyLoss** : la perte standard **torch.nn.CrossEntropyLoss**, sans poids de classes ni lissage. Simple et efficace quand les classes sont à peu près équilibrées ou quand on ne veut pas toucher à la loss.

**FocalLoss** : implémentée dans **optim.py**. Elle repose sur la cross-entropy mais pondère chaque exemple par **(1 - p_t)^gamma** : les exemples “faciles” (haute probabilité pour la bonne classe) contribuent moins, ce qui **focalise** l’entraînement sur les exemples difficiles. **gamma** est configurable — souvent 2,0. On peut activer **label_smoothing** et **use_class_weights** : dans ce cas **alpha** est le vecteur des poids de classes (inverse de la racine des effectifs), ce qui renforce encore l’attention sur les classes rares. Le choix de la FocalLoss est donc motivé par le déséquilibre des classes et la présence de nombreux exemples “faciles” qu’on ne veut pas dominer la descente de gradient.

---

## 6. Optimiseur et scheduler (≈ 3 min)

### 6.1 Optimiseur

**optim.build_optimizer(config, model, has_backbone)** adapte l’optimiseur au type de modèle.

**Avec backbone** (ResNet, EfficientNet, ConvNeXt, HfModel) : on utilise **AdamW** avec **deux groupes de paramètres** — backbone et tête — et des learning rates distincts : **backbone_lr** (ex. 1e-5 ou 3e-5) et **finetune_head_lr** (ex. 1e-4 ou 3e-4), avec un **weight_decay** commun (souvent 0,05). Les paramètres du backbone sont identifiés soit par **model.base_model** (HfModel), soit par l’exclusion de **fc** (ResNet) ou **classifier** (EfficientNet, ConvNeXt).

**Sans backbone** (Linear, FFN, etc.) : soit on utilise l’ancien schéma avec **optim.params** et **optim.algo** dans la config, soit un AdamW unique avec **lr** et **weight_decay**.

### 6.2 Scheduler

**optim.build_scheduler** gère la décroissance du learning rate sur la phase de fine-tuning.

- **CosineAnnealingLR** (défaut) : **T_max** = nombre d’epochs de fine-tuning moins le warmup LR, **eta_min** configurable (ex. 1e-7).  
- **ReduceLROnPlateau** : **mode="max"** car on optimise le **macro F1** validation ; **factor** (ex. 0,5), **patience** (ex. 5), **min_lr** (via **eta_min**). Le step du scheduler est alors appelé avec **test_metrics['f1']** après chaque epoch.

On peut ajouter un **finetune_warmup** : un nombre d’epochs en début de phase 2 où le LR est augmenté linéairement de 0 jusqu’aux valeurs cibles, avant d’enchaîner avec le scheduler principal. Cela évite des mises à jour trop agressives au tout début du fine-tuning.

---

## 7. Boucle d’entraînement (≈ 4 min)

### 7.1 Phases : warm-up puis fine-tuning

L’entraînement se fait en **deux phases** quand **warmup_epochs** est strictement positif et que le modèle a un backbone.

**Phase 1 — Warm-up** : pendant **warmup_epochs** (ex. 5), le **backbone est gelé** — **requires_grad=False** sur tous les paramètres sauf la tête (fc ou classifier). Seul un optimiseur AdamW sur les paramètres libres est utilisé, avec **head_lr** (ex. 1e-3). Objectif : entraîner rapidement la tête sur les features du backbone sans dégrader ces features. Si on **reprend** depuis un checkpoint (**resume.checkpoint**), cette phase est sautée pour ne pas réinitialiser la tête.

**Phase 2 — Fine-tuning** : tous les paramètres sont dégelés. On utilise **build_optimizer** (deux groupes si backbone) et **build_scheduler**. Les epochs restantes sont **nepochs - warmup_epochs**. C’est là qu’on applique le scheduler (CosineAnnealing ou ReduceLROnPlateau) et éventuellement le **finetune_warmup** du LR.

### 7.2 Mixed precision (AMP) et gradient clipping

Si **use_amp** est true et que CUDA est disponible, on utilise **torch.amp.autocast** et **GradScaler** pour l’entraînement en **mixed precision** : calculs en float16 quand c’est possible, avec mise à l’échelle des gradients pour éviter les underflows. Après **scaler.unscale_(optimizer)**, on applique **clip_grad_norm_(model.parameters(), max_norm=1.0)** pour limiter les explosions de gradient, puis **scaler.step** et **scaler.update**. C’est à la fois plus rapide et plus stable sur GPU.

### 7.3 Mixup et CutMix

La config peut définir **mix.mixup_alpha**, **mix.cutmix_alpha** et **mix.prob**. Si au moins un des alpha est strictement positif, à chaque batch on applique avec probabilité **mix.prob** soit **CutMix** (si cutmix_alpha > 0), soit **Mixup** (si mixup_alpha > 0). La loss est alors **lam * loss(targets_a) + (1 - lam) * loss(targets_b)** avec **lam** tiré d’une loi Beta. Mixup et CutMix augmentent la diversité des exemples et améliorent souvent la généralisation ; **prob** à 0,5 signifie qu’environ la moitié des batches sont mélangés.

### 7.4 Métriques et checkpointing

À chaque epoch on calcule **loss**, **accuracy** et **macro F1** (sklearn) sur le train et la validation. Le **meilleur modèle** est sauvegardé selon le **macro F1 validation** (higher is better) via **ModelCheckpoint** : à chaque amélioration du F1, on écrit **best_model.pt** (uniquement **model.state_dict()**). Tous les **checkpoint_every** epochs — et à la réception d’un SIGTERM — on sauvegarde **last_checkpoint.pt** avec **epoch**, **model_state_dict**, **optimizer_state_dict**, **scheduler_state_dict** pour pouvoir reprendre l’entraînement ou analyser la fin de run.

La config est sauvegardée dans **logdir/config.yaml**, et un **summary.txt** contient la commande, la config, le torchinfo du modèle et les infos sur les datasets. TensorBoard enregistre **Loss/Train**, **Loss/Validation**, **MacroF1/Validation**, **LR/Backbone**, **LR/Head**.

---

## 8. Reprise (resume) et inférence (≈ 2 min)

### 8.1 Reprise depuis un checkpoint

Dans la config, **resume.checkpoint** peut pointer vers un fichier **.pt**. Au démarrage de **train**, si ce chemin existe, on charge le checkpoint : soit c’est un dictionnaire contenant **model_state_dict**, soit c’est directement le state_dict. On charge dans le modèle (sans optimizer ni scheduler), ce qui permet de reprendre un entraînement ou de continuer à partir de poids pré-entraînés (par exemple un ResNet50 déjà entraîné par un collègue). Si on reprend, la phase de warm-up est désactivée.

### 8.2 Inférence simple et ensemble

**python -m torchtmpl.predict <config.yaml> <chemin_vers_best_model.pt>** : charge la config, construit le modèle avec **num_classes** (config ou 86), charge les poids depuis le checkpoint, lit les images dans **root_dir/test/imgs** avec les mêmes **img_size** et **resize_size** que l’entraînement, et écrit les prédictions (imgname, label) dans **submission.csv**, triées par nom de fichier.

**python -m torchtmpl.predict_ensemble cfg1 ckpt1 cfg2 ckpt2 ...** : charge plusieurs (config, checkpoint), fait les prédictions avec le même batch_size, puis combine en faisant la **somme des softmax** et **argmax** pour obtenir la classe finale. Les résultats sont écrits dans **submission_ensemble_Nmodels.csv**. C’est utile pour moyenner plusieurs modèles (par exemple ResNet50 + ConvNeXt) et améliorer la robustesse.

---

## 9. Déploiement sur le cluster (SLURM) (≈ 1 min 30)

**submit-slurm.py** prend en argument un fichier de config et optionnellement le nombre de runs. Il vérifie que le dépôt git est propre, copie la config dans **configs/** avec un nom temporaire, génère **job.sbatch** et lance **sbatch**. Le job SLURM utilise la partition **gpu_prod_long**, une limite de temps (ex. 12 h), et un job array si plusieurs runs sont demandés. Sur le nœud de calcul : copie du code (rsync), checkout du commit courant, **lien symbolique** de **logs** vers le répertoire permanent de l’utilisateur pour que les checkpoints et TensorBoard soient écrits directement sur le disque persistant, activation du venv, **pip install .**, puis **python -m torchtmpl.main <config> train**. À la fin, un rsync rapatrie le contenu de **logs** vers le répertoire de travail. Ainsi, toute l’expérimentation est reproductible et tracée (config + commit + logs).

---

## 10. Récapitulatif des choix et conclusion (≈ 1 min)

En résumé : **données** — ImageFolder, split stratifié 80/20, WeightedRandomSampler avec **sampler_power** 0,5, augmentations géométriques et colorimétriques + RandomErasing, normalisation ImageNet ; **modèles** — ResNet50, EfficientNet, ConvNeXt ou HfModel avec tête adaptée au nombre de classes ; **loss** — CrossEntropy ou FocalLoss avec poids de classes pour gérer le déséquilibre ; **optimisation** — AdamW à deux groupes (backbone / tête), scheduler CosineAnnealing ou ReduceLROnPlateau, warmup optionnel du head puis fine-tuning ; **entraînement** — AMP, Mixup/CutMix, gradient clipping, sauvegarde du meilleur modèle selon le macro F1 et du dernier checkpoint pour reprise ; **inférence** — predict single ou ensemble pour produire les fichiers de soumission. Chaque choix est piloté par la config YAML et documenté dans les logs de chaque run. Merci de m’avoir suivi, n’hésitez pas si vous avez des questions.

---

*Fin du script — durée totale visée : environ 20 minutes.*
