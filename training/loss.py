import torch
import torch.nn.functional as F

def contrastive_loss(hx, hy, temperature=0.1):

    similarity = torch.matmul(hx, hy.T) / temperature

    labels = torch.arange(hx.size(0)).to(hx.device)

    loss_i = F.cross_entropy(similarity, labels)
    loss_j = F.cross_entropy(similarity.T, labels)

    return (loss_i + loss_j) / 2