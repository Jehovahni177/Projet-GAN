## Generative Adversarial Network (GAN) – Swiss Roll 2D

# Description du projet

Ce projet implémente un GAN (Generative Adversarial Network) simple en PyTorch afin d’apprendre la distribution d’un jeu de données artificiel appelé Swiss Roll.

L’objectif est d’entraîner un générateur à produire des points 2D réalistes qui imitent la forme des données réelles, tandis qu’un discriminateur apprend à distinguer les points réels des points générés.

---

# Jeu de données : Swiss Roll 2D

- Les données sont générées à l’aide de sklearn.datasets.make_swiss_roll,

- Le Swiss Roll est initialement en 3D. Seules deux dimensions (x, z) sont conservées pour obtenir un problème en 2D,

- Le dataset contient 10 000 points avec du bruit aléatoire.

---

# Bibliothèques

Exécuter depuis le terminal de VS Code :

```
pip install torch torchvision torchaudio
pip install scikit-learn
pip install matplotlib
```

---

# Utilisation du GPU

Le code suivant détecte automatiquement la disponibilité d’un GPU :

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

GPU utilisé si disponible

Sinon, exécution sur CPU.

---

# Exécution

Il suffit de lancer le script Python. Via le terminal de VS Code, faites :

```
python gan_swiss_roll.py
```

Les graphiques s’afficheront automatiquement à la fin de l’entraînement.

---

# Architecture du GAN

**Générateur (Generator)** : Transformer du bruit aléatoire en points qui ressemblent aux données réelles.

- Entrée : bruit aléatoire de dimension 2,

- Réseau : perceptron multicouche (MLP),

- Sortie : point 2D généré.


**Discriminateur (Discriminator)** : Déterminer si un point provient du dataset réel ou du générateur.

- Entrée : point 2D (réel ou généré),

- Sortie : probabilité entre 0 et 1,

proche de 1 → point réel

proche de 0 → point généré

---

# Entraînement

**Principe**

Le GAN est entraîné de manière antagoniste :

1) Le générateur produit des faux points à partir de bruit aléatoire,

2) Le discriminateur apprend à distinguer les vrais des faux points,

3) Le générateur s’améliore pour tromper le discriminateur,

4) Le processus est répété sur plusieurs époques (nombre de passages complets sur toutes les données).

---

# Fonction de perte

```
BCELoss (Binary Cross Entropy Loss) adaptée à une classification binaire (réel = 1, faux = 0).
```
---

# Optimisation

- Optimiseur : Adam. Il permet d'ajuster les paramètres du réseau efficacement,

- Learning rate : Il représente la vitesse d’apprentissage (si trop grand alors instable, si trop petit alors lent),

- betas : Ils sont utilisés dans le but de contrôler la mémoire des directions de gradient et la stabilité de l’apprentissage. Les betas classiques d'un GAN sont betas=(0.5, 0.999). Ils permettent d’améliorer la stabilité de l’entraînement.

---

# Résultats et visualisation

Après l’entraînement, le générateur est utilisé pour produire 1000 nouveaux points. Deux visualisations sont affichées.

Celle du nuage de points des données générées et celle de l'histogramme 2D montrant la densité des points générés.

Ces graphiques permettent de vérifier visuellement si les points générés reproduisent bien la forme du Swiss Roll réel.

---

## Auteur

**Jéhovahni SODJINOU**  
Étudiant UTT - Mastère Spécialisé Expert Big Data Engineer

## Licence

Ce projet est réalisé à des fins éducatives pour l'UTT.

---

**Dernière mise à jour :** Février 2026