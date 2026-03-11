import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
import cv2

ground_truth_dir = "data/test/masks"          #  ground truth masks
pred_mask_dir = "predicted_masks"             # predicted masks
pred_uncertainty_dir = "predicted_uncertainty"  # predicted uncertainty maps
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

def dice_score(y_true, y_pred, eps=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2 * intersection + eps) / (np.sum(y_true) + np.sum(y_pred) + eps)

def expected_calibration_error(confidences, errors, n_bins=10):
    """ECE: average difference between confidence and accuracy per bin"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if np.sum(bin_mask) == 0:
            continue
        accuracy = np.mean(1 - errors[bin_mask])  
        avg_conf = np.mean(confidences[bin_mask])
        ece += np.sum(bin_mask) / len(confidences) * np.abs(avg_conf - accuracy)
    return ece

gt_files = sorted(os.listdir(ground_truth_dir))
pred_mask_files = sorted(os.listdir(pred_mask_dir))
pred_unc_files = sorted(os.listdir(pred_uncertainty_dir))

dice_scores = []
correlations = []
mi_scores = []
ece_scores = []

print("Starting visualization and evaluation...")

for gt_file, pred_file, unc_file in zip(gt_files, pred_mask_files, pred_unc_files):
    gt = cv2.imread(os.path.join(ground_truth_dir, gt_file), 0) / 255.0
    pred_mask = cv2.imread(os.path.join(pred_mask_dir, pred_file), 0) / 255.0
    pred_unc = cv2.imread(os.path.join(pred_uncertainty_dir, unc_file), 0) / 255.0

    # Dice
    dice = dice_score(gt, pred_mask)
    dice_scores.append(dice)

    # Correlation between uncertainty and error (pixel-wise)
    error_map = np.abs(gt - pred_mask)
    if np.std(pred_unc) == 0:
        corr = 0
    else:
        corr, _ = pearsonr(pred_unc.flatten(), error_map.flatten())
    correlations.append(corr)

    # MI
    mi = mutual_info_score(gt.flatten(), pred_unc.flatten())
    mi_scores.append(mi)

    # ECE
    ece = expected_calibration_error(pred_unc.flatten(), error_map.flatten(), n_bins=10)
    ece_scores.append(ece)


print(f"Average Dice Score: {np.mean(dice_scores):.4f}")
print(f"Average Correlation (Uncertainty vs Error): {np.mean(correlations):.4f}")
print(f"Average Mutual Information: {np.mean(mi_scores):.4f}")
print(f"Average ECE: {np.mean(ece_scores):.4f}")

for i in range(min(5, len(gt_files))):
    gt = cv2.imread(os.path.join(ground_truth_dir, gt_files[i]), 0) / 255.0
    pred_mask = cv2.imread(os.path.join(pred_mask_dir, pred_mask_files[i]), 0) / 255.0
    pred_unc = cv2.imread(os.path.join(pred_uncertainty_dir, pred_unc_files[i]), 0) / 255.0

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Uncertainty")
    plt.imshow(pred_unc, cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"example_{i}.png"))
    plt.show()

print("Visualization completed. Plots saved to 'results/' folder.")