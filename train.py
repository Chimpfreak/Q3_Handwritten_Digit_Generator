"""
Conditional Variational Autoencoder (CVAE) Training Script for MNIST Digits
Trains a class-conditional VAE to generate diverse handwritten digits (0–9).
Outputs: mnist_cvae.pth (PyTorch weights for app deployment)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Hyperparameters ---
batch_size = 128
epochs = 10            # 5–10 is enough for MNIST diversity
z_dim = 20             # Latent space size
num_classes = 10       # MNIST digit classes
img_size = 28          # MNIST image size

# --- Data Loading ---
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# --- Model Definition: Conditional VAE ---
class CVAE(nn.Module):
    """
    Conditional VAE model.
    - Encoder: Takes image and class embedding, outputs mean/logvar of latent z.
    - Decoder: Takes latent z and class embedding, outputs image.
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 10)
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(img_size*img_size + 10, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, z_dim)
        self.fc_logvar = nn.Linear(400, z_dim)
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, img_size*img_size),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        # x: [B, 1, 28, 28], c: [B]
        x = x.view(x.size(0), -1)
        c = self.embed(c)
        h = torch.cat([x, c], dim=1)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick for z sampling
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # z: [B, z_dim], c: [B]
        c = self.embed(c)
        h = torch.cat([z, c], dim=1)
        x_recon = self.decoder(h)
        return x_recon.view(-1, 1, img_size, img_size)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar

# --- Loss Function: VAE loss (BCE + KL) ---
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(imgs, labels)
        loss = loss_function(recon, imgs, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_data):.2f}')

# --- Save the trained model ---
torch.save(model.state_dict(), "mnist_cvae.pth")
print("Training done! Model saved to mnist_cvae.pth")
