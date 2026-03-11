import os
import cv2
import torch
from torch.utils.data import Dataset

class CamusDataset(Dataset):

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)

        mask_name = img_name.replace("frame", "mask")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)

        image = image / 255.0
        mask = mask / 255.0

        image = torch.tensor(image).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask