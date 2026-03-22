import torch
import cv2
import os
import numpy as np
from models.encoders import MaskEncoder
from torch.nn.functional import unfold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD =================
latent_vectors = torch.load("latent_space.pt").to(device)

encoder = MaskEncoder().to(device)
encoder.load_state_dict(torch.load("crisp_model.pth")["mask_encoder"])
encoder.eval()

pred_folder = "predicted_masks"
uncertainty_folder = "predicted_uncertainty"
os.makedirs(uncertainty_folder, exist_ok=True)

patch_size = 16
batch_size = 64  #  speed boost


# ================= PROCESS =================
for file in os.listdir(pred_folder):

    path = os.path.join(pred_folder, file)
    mask = cv2.imread(path, 0) / 255.0

    H, W = mask.shape
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():

        # -------- Extract patches --------
        patches = unfold(mask_tensor, kernel_size=patch_size, stride=patch_size)
        patches = patches.permute(0, 2, 1)  # [1, N, patch_dim]
        patches = patches.reshape(-1, 1, patch_size, patch_size)  # [N,1,16,16]

        # -------- Batch encoding --------
        embeddings = []
        for i in range(0, patches.shape[0], batch_size):
            batch = patches[i:i+batch_size]
            z = encoder(batch)
            embeddings.append(z)

        patch_embeddings = torch.cat(embeddings, dim=0)  # [N, latent_dim]

        # -------- Distance computation --------
        distances = torch.cdist(patch_embeddings, latent_vectors)
        min_distances = distances.min(dim=1)[0]

    # ================= RECONSTRUCT =================
    uncertainty_map = torch.zeros((H, W))

    idx = 0
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            h_end = min(i + patch_size, H)
            w_end = min(j + patch_size, W)

            uncertainty_map[i:h_end, j:w_end] = min_distances[idx].cpu()
            idx += 1

    # ================= NORMALIZE =================
    min_val = uncertainty_map.min()
    max_val = uncertainty_map.max()

    if max_val - min_val > 1e-8:
        uncertainty_map = (uncertainty_map - min_val) / (max_val - min_val)
    else:
        uncertainty_map = torch.zeros_like(uncertainty_map)

    # ================= SAVE =================
    save_path = os.path.join(
        uncertainty_folder,
        file.replace("pred_", "uncertainty_")
    )

    cv2.imwrite(save_path, (uncertainty_map.numpy() * 255).astype(np.uint8))

    print(f"{file} processed. Uncertainty map saved.")
