# CRISP-Based Uncertainty Estimation for Medical Image Segmentation
## Overview:
This project implements uncertainty estimation for medical image segmentation using CRISP (Contrastive Representation for Image Segmentation Prediction) with a U-Net segmentation model.
The goal is to improve model reliability in medical imaging by identifying regions where the model is uncertain about its predictions.

## Pipeline:
Medical Image
      ↓
U-Net Segmentation
      ↓
Predicted Mask
      ↓
CRISP Latent Space
      ↓
Uncertainty Estimation
      ↓
Evaluation & Visualization

## Example Output
Below are some examples from the model:
![alt text](Figure_1.png)
<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/5fd7d7be-8ebd-4142-8a2e-78d21611a6ea" />
<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/068f6157-f619-4220-9ad7-70a3dd7b1920" />
<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/5822f701-66d8-4de3-b87a-394cd4b6834f" />
<img width="1200" height="400" alt="image" src="https://github.com/user-attachments/assets/094a7232-58cb-4312-8c6d-9e0d1d9cbefe" />


Left: Ground truth mask
Middle: Predicted mask from U-Net
Right: Uncertainty heatmap from CRISP
Bright regions indicate high uncertainty.

## Project structure
CRISP_Project/
│
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   │
│   └── test/
│       ├── images/
│       └── masks/
│
├── models/
│   ├── unet.py
│   |── encoders.py
|   └──projection.py
│
├── utils/
│   └── dataset.py
|
|── training/
|    └──train.py
│    └──loss.py
|
├── predicted_masks/
├── predicted_uncertainty/
├── results/
│
├── train_unet.py
├── build_latent_space.py
├── predict_masks.py
├── predict_uncertainty.py
├── visualize_and_evaluate.py
│
└── README.md

## Installation
Clone the repository:
git clone https://github.com/YOUR_USERNAME/crisp-uncertainty-segmentation.git
cd crisp-uncertainty-segmentation

Install dependencies:
pip install torch torchvision numpy opencv-python matplotlib scikit-learn

## Training the Segmentation Model

Train the U-Net model:

python train_unet.py

Output:

unet_model.pth
Building CRISP Latent Space

Run:

python build_latent_space.py

Output:

latent_space.pt
crisp_model.pth
Predict Segmentation Masks
python predict_masks.py

Output:

predicted_masks/
Compute Uncertainty Maps
python predict_uncertainty.py

Output:

predicted_uncertainty/
Evaluation and Visualization

Run:

python visualize_and_evaluate.py

This computes:

Dice Score

Uncertainty vs Error Correlation

Mutual Information

Expected Calibration Error (ECE)

Results and visualizations are saved in:

results/
Example Metrics

From current experiments:

Average Dice Score: 0.8354
Average Correlation (Uncertainty vs Error): 0.3040
Average Mutual Information: 0.7863
Average ECE: 0.2458

The same training pipeline has been tried on Covid-19 CT scan dataset
Dataset link: https://www.kaggle.com/datasets/nguyentienda32143/covid-19-ct-lung-and-infection-segmentation
Metrics:
Correlation : 0.2468
Mutual Information: 0.6234
ECE: 0.2312
Some visualizations:
<img width="1660" height="453" alt="image" src="https://github.com/user-attachments/assets/bb7331b9-a799-4f6e-9ad1-a8b0e1764abd" />
<img width="1646" height="450" alt="Screenshot 2026-03-21 124832" src="https://github.com/user-attachments/assets/554a7ab8-639d-4938-a737-21a1958b402d" />
<img width="1658" height="496" alt="Screenshot 2026-03-21 124818" src="https://github.com/user-attachments/assets/73e46312-0d7d-4d49-8987-1f4b1b204b2c" />
<img width="1635" height="460" alt="Screenshot 2026-03-21 124300" src="https://github.com/user-attachments/assets/934bab52-394f-4d33-bb2f-9a7d2a9c3644" />





