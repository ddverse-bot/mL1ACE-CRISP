import torch
from train_unet import UNet
from utils.dataset import CamusDataset
from torch.utils.data import DataLoader
import cv2
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
dataset = CamusDataset("data/test/images", "data/test/masks")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# model
model = UNet().to(device)
model.load_state_dict(torch.load("unet_model.pth", map_location=device))
model.eval()

#  output folder
os.makedirs("predicted_masks", exist_ok=True)

# predict and save masks
with torch.no_grad():
    for i, (img, _) in enumerate(loader):
        img = img.to(device)
        pred = model(img)
        pred = pred.squeeze().cpu().numpy()
        # save mask
        cv2.imwrite(f"predicted_masks/mask_{i:04d}.png", (pred * 255).astype("uint8"))

print("All predicted masks saved in 'predicted_masks/'")