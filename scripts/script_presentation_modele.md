# Script vidéo — Présentation du modèle VanillaCNN

**Durée indicative : 3–5 minutes**

---

## 1. Introduction (30 s)

« Bonjour, je vais vous présenter l’architecture du modèle que j’ai conçu pour la classification de plancton : une CNN que j’ai nommée **VanillaCNN**. Je vais détailler les choix d’architecture et les raisons derrière chacun. »

---

## 2. Contexte et objectif (20 s)

« Le modèle doit prendre en entrée des images (canaux, hauteur, largeur), les traiter par des couches convolutives, et produire en sortie un score par classe pour la classification. L’architecture est **configurable** : on peut faire varier le nombre de blocs via un fichier de config, ce qui permet d’adapter la profondeur du réseau sans toucher au code. »

---

## 3. Blocs de base (1 min)

« J’ai factorisé l’architecture en deux types de blocs réutilisables.

**Premier bloc : `conv_relu_bn`.**  
C’est une séquence **convolution 3×3 → ReLU → BatchNorm**. J’ai choisi des convolutions **3×3** parce qu’elles capturent bien des motifs locaux sans exploser le nombre de paramètres, et qu’on peut empiler plusieurs couches pour avoir un champ récepteur plus grand. Le **ReLU** apporte la non-linéarité. La **BatchNorm** stabilise l’apprentissage et accélère la convergence, ce qui est utile quand on empile plusieurs couches.

**Deuxième bloc : `conv_down`.**  
Ici, au lieu d’un classique MaxPooling, j’ai utilisé une **convolution 2×2 avec stride 2** pour faire la réduction de résolution. C’est ce qu’on appelle du **downsampling convolutif** : les poids sont appris, donc le réseau peut choisir comment résumer l’information spatiale, plutôt que de prendre simplement le maximum comme en MaxPool. Après la conv, on a encore ReLU et BatchNorm pour rester cohérent avec le reste du réseau. »

---

## 4. Structure en blocs répétés (1 min)

« Le cœur du modèle est une **boucle sur le nombre de blocs** défini dans la config.

Dans chaque bloc, je fais **deux fois** la séquence conv_relu_bn : une fois pour passer du nombre de canaux d’entrée au nombre de canaux du bloc, une deuxième fois en restant sur le même nombre de canaux. Ça donne deux couches 3×3 consécutives par bloc, ce qui augmente un peu la profondeur et la capacité sans changer la résolution. Ensuite, j’applique **conv_down** : la résolution est divisée par deux et le **nombre de canaux est doublé**. C’est un schéma classique en CNN : plus on descend en résolution, plus on augmente le nombre de feature maps pour compenser.

Au fil des blocs, la taille spatiale diminue et la dimension des canaux augmente, ce qui permet de passer de détails locaux en début de réseau à des représentations plus globales en fin de réseau. »

---

## 5. Sortie : Global Average Pooling (45 s)

« À la fin des blocs convolutifs, au lieu d’un gros empilement de couches fully connected, j’ai mis un **Adaptive Average Pooling 2D** vers la taille 1×1. Concrètement, pour chaque canal, on fait la moyenne sur toute la carte de features. On obtient donc un vecteur de taille “nombre de canaux”, qu’on aplatit puis qu’on envoie dans **une seule couche linéaire** vers le nombre de classes.

Ce choix limite fortement le nombre de paramètres en fin de réseau et réduit les risques d’overfitting, tout en gardant une information résumée par canal. C’est une idée qu’on retrouve dans des architectures comme GoogLeNet ou ResNet. »

---

## 6. Récapitulatif des choix (30 s)

« Pour résumer les choix d’architecture :

- **Convolutions 3×3** pour l’extraction de features, avec ReLU et BatchNorm.
- **Downsampling par convolution 2×2 stride 2** plutôt que MaxPool, pour un sous-échantillonnage appris.
- **Structure en blocs répétables** : deux conv 3×3 puis une conv_down, avec doublement des canaux à chaque bloc.
- **Global Average Pooling** et une seule couche linéaire en sortie pour garder le modèle léger et stable.

Le tout est implémenté de façon modulaire et piloté par une config, ce qui permet d’ajuster facilement la profondeur du réseau selon la complexité des données. Merci de votre attention. »

---

## Notes pour la vidéo

- **Montrer le code** : afficher `cnn_models.py` pendant les parties 3, 4 et 5 (blocs, boucle, GAP + Linear).
- **Schéma** : un petit schéma “entrée → blocs (conv, conv, down) × N → GAP → Linear → sortie” peut aider.
- **Ton** : posé et pédagogique ; insister sur le “pourquoi” de chaque choix (stabilité, paramètres, flexibilité).
