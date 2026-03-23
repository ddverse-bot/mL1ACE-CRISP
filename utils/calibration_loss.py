import torch
import torch.nn as nn


class HardL1ACELoss(nn.Module):
    def __init__(self, n_bins=20, eps=1e-8):
        super().__init__()
        self.n_bins = n_bins
        self.eps = eps

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        bins = torch.linspace(0, 1, self.n_bins + 1, device=preds.device)

        ace = 0.0
        valid_bins = 0

        for i in range(self.n_bins):
            mask = (preds >= bins[i]) & (preds < bins[i + 1])

            if mask.sum() == 0:
                continue

            conf = preds[mask].mean()
            acc = targets[mask].float().mean()

            ace += torch.abs(conf - acc)
            valid_bins += 1

        if valid_bins > 0:
            ace = ace / valid_bins

        return ace


class SoftL1ACELoss(nn.Module):
    def __init__(self, n_bins=20, eps=1e-8):
        super().__init__()
        self.n_bins = n_bins
        self.eps = eps

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        bin_centers = torch.linspace(0, 1, self.n_bins, device=preds.device)

        ace = 0.0

        for c in bin_centers:
            # triangular soft weighting
            weights = torch.clamp(1 - torch.abs(preds - c) * self.n_bins, min=0)

            if weights.sum() < self.eps:
                continue

            conf = (weights * preds).sum() / (weights.sum() + self.eps)
            acc = (weights * targets).sum() / (weights.sum() + self.eps)

            ace += torch.abs(conf - acc)

        return ace / self.n_bins
