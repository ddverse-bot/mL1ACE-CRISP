import torch
from models.unet import UNet   
from utils.dataset import CamusDataset
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= DATA =================
dataset = CamusDataset("data/test/images", "data/test/masks")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ================= MODEL =================
model = UNet().to(device)
model.load_state_dict(torch.load("best_unet_model.pth", map_location=device))
model.eval()

# ================= OUTPUT FOLDERS =================
os.makedirs("predicted_masks", exist_ok=True)
os.makedirs("predicted_probs", exist_ok=True)  

print("Starting inference...")

with torch.no_grad():
    for i, (img, _) in enumerate(loader):

        img = img.to(device)

        # -------- Forward --------
        pred = model(img)
        prob = torch.sigmoid(pred)   

        prob_np = prob.squeeze().cpu().numpy()

        # -------- Binary mask --------
        mask = (prob_np > 0.5).astype(np.float32)

        # -------- Save --------
        cv2.imwrite(
            f"predicted_masks/mask_{i:04d}.png",
            (mask * 255).astype("uint8")
        )

        cv2.imwrite(
            f"predicted_probs/prob_{i:04d}.png",
            (prob_np * 255).astype("uint8")
        )

        print(f"Saved sample {i}")

print("\nAll predictions saved!")
