import torch
import torch.nn as nn


def compute_ace(preds, targets, num_bins=20, eps=1e-8):
    """
    preds: probabilities (after sigmoid) -> shape (B, 1, H, W)
    targets: ground truth masks -> shape (B, 1, H, W)
    """

    preds = preds.view(-1)
    targets = targets.view(-1)

    bins = torch.linspace(0, 1, num_bins + 1, device=preds.device)
    ace = 0.0
    valid_bins = 0

    for i in range(num_bins):
        mask = (preds >= bins[i]) & (preds < bins[i+1])

        if mask.sum() == 0:
            continue

        e = preds[mask].mean()              # confidence
        o = targets[mask].float().mean()    # accuracy

        ace += torch.abs(e - o)
        valid_bins += 1

    return ace / (valid_bins + eps)


# ---------------- HARD mL1-ACE ----------------
class HardL1ACELoss(nn.Module):
    def __init__(self, num_bins=20):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, preds, targets):
        return compute_ace(preds, targets, self.num_bins)


# ---------------- SOFT mL1-ACE ----------------
class SoftL1ACELoss(nn.Module):
    def __init__(self, num_bins=20, temperature=0.1):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature

    def forward(self, preds, targets):
        """
        Soft binning using Gaussian weights
        """
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        bin_centers = torch.linspace(0, 1, self.num_bins, device=preds.device)

        ace = 0.0

        for c in bin_centers:
            weights = torch.exp(-((preds_flat - c) ** 2) / self.temperature)

            if weights.sum() == 0:
                continue

            e = (weights * preds_flat).sum() / (weights.sum() + 1e-8)
            o = (weights * targets_flat).sum() / (weights.sum() + 1e-8)

            ace += torch.abs(e - o)

        return ace / self.num_bins
