import torch
from utils.dataset import CamusDataset
from models.encoders import MaskEncoder
from torch.utils.data import DataLoader

dataset = CamusDataset("data/train/images", "data/train/masks")
loader = DataLoader(dataset, batch_size=16)

encoder = MaskEncoder()
encoder.load_state_dict(torch.load("crisp_model.pth")["mask_encoder"])
encoder.eval()

latent_vectors = []

with torch.no_grad():

    for img, mask in loader:

        z = encoder(mask)
        latent_vectors.append(z)

latent_vectors = torch.cat(latent_vectors)

torch.save(latent_vectors, "latent_space.pt")

print("Latent space saved!")