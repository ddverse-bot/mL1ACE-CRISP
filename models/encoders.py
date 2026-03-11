import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):

    def __init__(self, latent_dim=128):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        z = self.fc(x)

        return z


class MaskEncoder(nn.Module):

    def __init__(self, latent_dim=128):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)

        z = self.fc(x)

        return z