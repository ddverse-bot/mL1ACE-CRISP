import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from utils.dataset import CamusDataset
from models.encoders import ImageEncoder, MaskEncoder
from models.projection import ProjectionHead
from training.loss import contrastive_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
train_dataset = CamusDataset(
    "data/train/images",
    "data/train/masks"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# Models
image_encoder = ImageEncoder().to(device)
mask_encoder = MaskEncoder().to(device)

proj_x = ProjectionHead().to(device)
proj_y = ProjectionHead().to(device)

# Optimizer
optimizer = torch.optim.Adam(
    list(image_encoder.parameters()) +
    list(mask_encoder.parameters()) +
    list(proj_x.parameters()) +
    list(proj_y.parameters()),
    lr=1e-3
)

epochs = 10

for epoch in range(epochs):

    total_loss = 0

    for img, mask in train_loader:

        img = img.to(device)
        mask = mask.to(device)

        zx = image_encoder(img)
        zy = mask_encoder(mask)

        hx = proj_x(zx)
        hy = proj_y(zy)

        loss = contrastive_loss(hx, hy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
    torch.save({
    "image_encoder": image_encoder.state_dict(),
    "mask_encoder": mask_encoder.state_dict(),
    "proj_x": proj_x.state_dict(),
    "proj_y": proj_y.state_dict()
}, "crisp_model.pth")

print("Model saved")