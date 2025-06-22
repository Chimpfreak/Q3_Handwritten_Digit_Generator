import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Model settings (must match training)
z_dim = 20
num_classes = 10
img_size = 28

# CVAE model (decoder only)
class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_classes, 10)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(img_size*img_size + 10, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, z_dim)
        self.fc_logvar = nn.Linear(400, z_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, img_size*img_size),
            nn.Sigmoid()
        )

    def decode(self, z, c):
        c = self.embed(c)
        h = torch.cat([z, c], dim=1)
        x_recon = self.decoder(h)
        return x_recon.view(-1, 1, img_size, img_size)


# Load trained weights
@st.cache_resource
def load_model():
    model = CVAE()
    model.load_state_dict(torch.load("mnist_cvae.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("Handwritten Digit Generator (MNIST style, CVAE)")
digit = st.selectbox("Select digit to generate", list(range(10)))

if st.button("Generate 5 images"):
    with torch.no_grad():
        z = torch.randn(5, z_dim)
        labels = torch.full((5,), digit, dtype=torch.long)
        imgs = model.decode(z, labels).cpu().numpy()
        imgs = (imgs * 255).astype(np.uint8)
    st.write(f"5 samples of digit: {digit}")
    st.image([img[0] for img in imgs], width=96, caption=[f"{digit} #{i+1}" for i in range(5)])
