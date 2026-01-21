import torch
import torch.nn as nn
from torch.optim import Adam
import sklearn.datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Utilisation du GPU si disponible

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Création du dataset (les données d'entraînement)
# On génère un dataset "Swiss Roll" (en forme de spirale en 3D) avec du bruit dont on ne garde que 2 dimensions (x, z)

class SR_Dataset(Dataset):
    def __init__(self):
        srx, sry = sklearn.datasets.make_swiss_roll(n_samples=10000, noise=0.3, random_state=None)
        srx = torch.tensor(srx, dtype=torch.float) / 10
        self.x_train = torch.cat((srx[:, 0].reshape(-1, 1), srx[:, 2].reshape(-1, 1)), 1)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx]

    def get_data(self):
        return self.x_train


# Un GAN (Generative Adversarial Network) contient deux réseaux antagonistes, c'est-à-dire,
# jouant l’un contre l’autre. Il s'agit d'un Générateur (Generator) fabriquant, par exemple, des faux points qui ressemblent aux vrais 
# et d'un Discriminateur (Discriminator) essayant de deviner si un point est vrai ou faux.
# Le générateur s'améliore pour tromper le discriminateur, et le discriminateur s'améliore pour mieux détecter.
# À la fin, les faux points deviennent très réalistes.



# Generator

# Le générateur prend en entrée du bruit aléatoire (2 nombres au hasard),
# et via un réseau de neurones (MLP) transforme ce bruit en un point 2D qui doit ressembler aux points réels du dataset


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.model(x)



# Discriminator

# Le discriminateur prend un point 2D et renvoie une probabilité entre 0 et 1. 
# Si c'est proche de 1 alors il déduit que c'est un vrai point (du coup, provenant du dataset)
# Si c'est proche de 0 alors il déduit que c'est un faux point (du coup, provenant du générateur)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(2, 128), nn.LeakyReLU(0.2), nn.Linear(128, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


# Initialisation des modèles et optimisateurs 


# Initialisation des deux modèles (générateur et discriminateur), puis envoi sur le device (GPU/CPU)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Utilisation de l'optimisateur Adam

# Il permet d'ajuster les paramètres du réseau efficacement
# Le learning rate(lr) représente la vitesse d’apprentissage (si trop grand alors instable, si trop petit alors lent)
# Les betas sont utilisés dans le but de contrôler la mémoire des directions de gradient et la stabilité de l’apprentissage.
# Les betas classiques d'un GAN sont betas=(0.5, 0.999).

optim_g = Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optim_d = Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))


# Préparation des données

dataset = SR_Dataset()

# Le DataLoader sert à récupérer les données par petits lots (batch) au lieu de tout prendre d’un coup.
# Cela est plus rapide et plus stable

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)


# Entraînement


# num_epochs = nombre de passages complets sur toutes les données

num_epochs = 5

# BCELoss = Binary Cross Entropy Loss
# C'est une mesure d'erreur utilisée dans le cadre d'une classification binaire : vrai (1) ou faux (0)

criterion = nn.BCELoss()

num_epochs = 5
for epoch in range(num_epochs):
    for real_samples in dataloader:

        # On récupère des points réels en 2D et on les envoie sur GPU/CPU

        real_samples = real_samples.to(device)          

        # Génération des échantillons
        # On génère du bruit aléatoire (autant d'exemples que dans le batch)

        noise = torch.randn(real_samples.size(0), 2, device=device)

        # Le générateur transforme le bruit en faux points 2D

        fake_samples = generator(noise)


        # Entraînement du discriminateur


        optim_d.zero_grad()

        real_labels = torch.ones(real_samples.size(0), 1, device=device)
        fake_labels = torch.zeros(real_samples.size(0), 1, device=device)

        out_real = discriminator(real_samples)
        out_fake = discriminator(fake_samples.detach())

        loss_real = criterion(out_real, real_labels)
        loss_fake = criterion(out_fake, fake_labels)

        # Erreur totale du discriminateur

        loss_d = (loss_real + loss_fake) / 2

        # On corrige le discriminateur pour qu'il devienne meilleur

        loss_d.backward()
        optim_d.step()


        # Entraînement du générateur


        optim_g.zero_grad()

        # On repasse les faux points dans le discriminateur (sans detach cette fois)

        out_fake_for_g = discriminator(fake_samples)

        loss_g = criterion(out_fake_for_g, real_labels)

        # On corrige le générateur pour qu'il devienne davantage meilleur pour tromper le discriminateur

        loss_g.backward()
        optim_g.step()

    # Affichage des pertes (erreurs) à la fin de chaque époque

    print(f"Epoch: {epoch} Loss D.: {loss_d}")
    print(f"Epoch: {epoch} Loss G.: {loss_g}")

# Validation (générer des points après l'entraînement)

# On met le générateur en mode évaluation

generator.eval()

# On génère un nouveau lot de bruit(1000), puis 1000 points faux

noise_samples = torch.randn(1000, 2, device=device)
generated_samples = generator(noise_samples)


# Visualisation (voir si les points générés "ressemblent" aux données réelles)

# Nuage de points (montre la forme globale des points générés)

generated_samples = generated_samples.cpu().detach().numpy()
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='r')
plt.title('Generated Samples for Validation')
plt.show()

# Histogramme 2D (densité) des points générés

plt.hist2d(generated_samples[:, 0], generated_samples[:, 1], bins=100)
plt.title('Histogram of Generated Samples')
plt.show()
