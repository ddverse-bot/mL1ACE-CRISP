import torch

def compute_ace(preds, targets, num_bins=20, eps=1e-8):
    """
    preds: (B, C, H, W) probabilities after softmax/sigmoid
    targets: (B, H, W) or (B, C, H, W)
    """

    # Flatten everything
    preds = preds.view(-1)
    targets = targets.view(-1)

    bins = torch.linspace(0, 1, num_bins + 1, device=preds.device)
    ace = 0.0

    for i in range(num_bins):
        mask = (preds >= bins[i]) & (preds < bins[i+1])

        if mask.sum() == 0:
            continue

        e = preds[mask].mean()              # confidence
        o = targets[mask].float().mean()    # accuracy

        ace += torch.abs(e - o)

    return ace / num_bins
