A PyTorch implementation of U-Net segmentation with CRISP uncertainty estimation, calibrated losses (ACE, ECE, MCE), and visualizations on medical imaging datasets.

 Features
U-Net segmentation for medical images
CRISP-based patch latent space encoder for uncertainty estimation
Calibration metrics:
Expected Calibration Error (ECE)
Average Calibration Error (ACE)
Maximum Calibration Error (MCE)
Additional metrics: Dice Score, Correlation (uncertainty vs error), Mutual Information
Visualizations:
Ground Truth vs Predicted Masks
Uncertainty maps
Error maps
Reliability diagrams
WandB integration for training tracking and logging
  Repository Structure
ddverse-bot/
│
├─ data/                    # Dataset folder
│  ├─ train/images/
│  ├─ train/masks/
│  ├─ test/images/
│  └─ test/masks/
│
├─ models/                  # U-Net and CRISP models
│  ├─ unet.py
│  └─ encoders.py
│
├─ utils/                   # Helper scripts
│  ├─ dataset.py
│  └─ calibration_loss.py
│
├─ predicted_masks/          # Saved predicted masks
├─ predicted_uncertainty/    # Saved uncertainty maps
├─ results/                  # Visualization outputs
│
├─ train_unet.py             # Training U-Net with calibration losses
├─ predict_masks.py          # Predict segmentation masks
├─ predict_uncertainty.py    # Generate CRISP uncertainty maps
├─ visualize_and_evaluate.py # Compute metrics and generate plots
├─ latent_space.pt           # Pretrained latent space vectors
├─ crisp_model.pth           # Pretrained CRISP encoder model
└─ README.md
 Installation
# Clone the repository
git clone https://github.com/<your_username>/ddverse-bot.git
cd ddverse-bot

# Create environment
conda create -n ddverse python=3.10
conda activate ddverse

# Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python matplotlib scikit-learn wandb
 Usage
1. Train U-Net
python train_unet.py
Logs metrics and losses to WandB
Saves best model as best_unet_model.pth
2. Predict Masks
python predict_masks.py
Outputs predicted masks to predicted_masks/
3. Predict Uncertainty
python predict_uncertainty.py
Uses CRISP encoder to generate uncertainty maps
Saves outputs to predicted_uncertainty/
4. Visualize & Evaluate
python visualize_and_evaluate.py
Computes Dice, Correlation, MI, ECE, ACE, MCE
Saves visualization plots to results/
 Metrics & Visualization
Dice Score: Segmentation overlap accuracy
Correlation (Uncertainty vs Error): How well uncertainty predicts errors
Mutual Information: Relationship between ground truth and uncertainty
ECE, ACE, MCE: Calibration metrics
Visualizations: GT, predicted mask, uncertainty, error map, reliability diagram
<img width="590" height="590" alt="image" src="https://github.com/user-attachments/assets/d42e45d1-b651-41fa-ad5f-8d5d9eb4be96" />

