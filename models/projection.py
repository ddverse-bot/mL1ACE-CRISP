import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):

    def __init__(self, latent_dim=128, proj_dim=64):
        super().__init__()

        self.linear = nn.Linear(latent_dim, proj_dim)

    def forward(self, z):

        h = self.linear(z)

        h = F.normalize(h, dim=1)

        return h