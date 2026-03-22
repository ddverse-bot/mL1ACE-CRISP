import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr
import cv2


ground_truth_dir = "data/test/masks"
pred_mask_dir = "predicted_masks"
pred_uncertainty_dir = "predicted_uncertainty"
results_dir = "results"

os.makedirs(results_dir, exist_ok=True)



def dice_score(y_true, y_pred, eps=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2 * intersection + eps) / (np.sum(y_true) + np.sum(y_pred) + eps)


def compute_calibration(confidences, errors, n_bins=10):
    """
    Computes:
    - ECE (Expected Calibration Error)
    - ACE (Average Calibration Error)
    - MCE (Maximum Calibration Error)
    """

    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    ace = 0.0
    mce = 0.0
    valid_bins = 0

    for i in range(n_bins):
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])

        if np.sum(bin_mask) == 0:
            continue

        accuracy = np.mean(1 - errors[bin_mask])
        avg_conf = np.mean(confidences[bin_mask])

        diff = np.abs(avg_conf - accuracy)

        # ECE (weighted)
        ece += np.sum(bin_mask) / len(confidences) * diff

        # ACE (unweighted)
        ace += diff
        valid_bins += 1

        # MCE
        mce = max(mce, diff)

    ace = ace / valid_bins if valid_bins > 0 else 0

    return ece, ace, mce


def reliability_diagram(confidences, errors, save_path):
    """Plot reliability diagram"""
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    accs = []
    confs = []

    for i in range(n_bins):
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])

        if np.sum(bin_mask) == 0:
            continue

        accuracy = np.mean(1 - errors[bin_mask])
        avg_conf = np.mean(confidences[bin_mask])

        accs.append(accuracy)
        confs.append(avg_conf)

    plt.figure()
    plt.plot(confs, accs, marker='o', label="Model")
    plt.plot([0, 1], [0, 1], linestyle='--', label="Perfect Calibration")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()

    plt.savefig(save_path)
    plt.close()



gt_files = sorted(os.listdir(ground_truth_dir))
pred_mask_files = sorted(os.listdir(pred_mask_dir))
pred_unc_files = sorted(os.listdir(pred_uncertainty_dir))



dice_scores = []
correlations = []
mi_scores = []
ece_scores = []
ace_scores = []
mce_scores = []


print("Starting evaluation...\n")


for gt_file, pred_file, unc_file in zip(gt_files, pred_mask_files, pred_unc_files):

    gt = cv2.imread(os.path.join(ground_truth_dir, gt_file), 0) / 255.0
    pred_mask = cv2.imread(os.path.join(pred_mask_dir, pred_file), 0) / 255.0
    pred_unc = cv2.imread(os.path.join(pred_uncertainty_dir, unc_file), 0) / 255.0

    # Binarize prediction (important!)
    pred_mask_bin = (pred_mask > 0.5).astype(np.float32)

    # Dice
    dice = dice_score(gt, pred_mask_bin)
    dice_scores.append(dice)

    # Error map
    error_map = np.abs(gt - pred_mask_bin)

    # Confidence = 1 - uncertainty
    confidence = 1 - pred_unc

    # Correlation (uncertainty vs error)
    if np.std(pred_unc) == 0:
        corr = 0
    else:
        corr, _ = pearsonr(pred_unc.flatten(), error_map.flatten())
    correlations.append(corr)

    # Mutual Information
    mi = mutual_info_score(gt.flatten(), pred_unc.flatten())
    mi_scores.append(mi)

    # Calibration metrics
    ece, ace, mce = compute_calibration(
        confidence.flatten(),
        error_map.flatten(),
        n_bins=10
    )

    ece_scores.append(ece)
    ace_scores.append(ace)
    mce_scores.append(mce)



print("----- FINAL RESULTS -----")
print(f"Average Dice Score: {np.mean(dice_scores):.4f}")
print(f"Avg Correlation (Uncertainty vs Error): {np.mean(correlations):.4f}")
print(f"Average Mutual Information: {np.mean(mi_scores):.4f}")
print(f"Average ECE: {np.mean(ece_scores):.4f}")
print(f"Average ACE: {np.mean(ace_scores):.4f}")
print(f"Average MCE: {np.mean(mce_scores):.4f}")



print("\nGenerating visualizations...")

for i in range(min(5, len(gt_files))):

    gt = cv2.imread(os.path.join(ground_truth_dir, gt_files[i]), 0) / 255.0
    pred_mask = cv2.imread(os.path.join(pred_mask_dir, pred_mask_files[i]), 0) / 255.0
    pred_unc = cv2.imread(os.path.join(pred_uncertainty_dir, pred_unc_files[i]), 0) / 255.0

    confidence = 1 - pred_unc
    error_map = np.abs(gt - (pred_mask > 0.5).astype(np.float32))

    # Plot images
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Prediction")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Uncertainty")
    plt.imshow(pred_unc, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Error Map")
    plt.imshow(error_map, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"example_{i}.png"))
    plt.close()

    # Reliability diagram
    reliability_diagram(
        confidence.flatten(),
        error_map.flatten(),
        os.path.join(results_dir, f"reliability_{i}.png")
    )


print(" Done! Results saved in 'results/' folder.")ed to 'results/' folder.")
