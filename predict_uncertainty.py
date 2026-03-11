import torch
import cv2
import os
import numpy as np
from models.encoders import MaskEncoder
from torch.nn.functional import unfold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_vectors = torch.load("latent_space.pt").to(device) 
encoder = MaskEncoder().to(device)
encoder.load_state_dict(torch.load("crisp_model.pth")["mask_encoder"])
encoder.eval()
pred_folder = "predicted_masks"                
uncertainty_folder = "predicted_uncertainty"  
os.makedirs(uncertainty_folder, exist_ok=True)


patch_size = 16  

for file in os.listdir(pred_folder):
    path = os.path.join(pred_folder, file)
    mask = cv2.imread(path, 0) / 255.0 
    H, W = mask.shape
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,H,W]

    with torch.no_grad():
        
        patches = unfold(mask_tensor, kernel_size=patch_size, stride=patch_size) 
        patches = patches.permute(0, 2, 1)  
        patch_embeddings = []
        for p in patches[0]:
            p = p.view(1, 1, patch_size, patch_size)  
            z = encoder(p)  
            patch_embeddings.append(z)
        patch_embeddings = torch.cat(patch_embeddings, dim=0)  
        distances = torch.cdist(patch_embeddings, latent_vectors)  
        min_distances = distances.min(dim=1)[0]  
    uncertainty_map = torch.zeros((H, W))
    idx = 0
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            h_end = min(i + patch_size, H)
            w_end = min(j + patch_size, W)
            uncertainty_map[i:h_end, j:w_end] = min_distances[idx].cpu()
            idx += 1
    uncertainty_map = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min() + 1e-8)
    save_path = os.path.join(uncertainty_folder, file.replace("pred_", "uncertainty_"))
    cv2.imwrite(save_path, (uncertainty_map.numpy() * 255).astype(np.uint8))

    print(f"{file} processed. Uncertainty map saved.")