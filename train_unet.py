import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.dataset import CamusDataset


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading dataset...")
    dataset = CamusDataset("data/train/images", "data/train/masks")
    print("Total images:", len(dataset))

    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    print("Initializing UNet...")
    model = UNet().to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    print("Starting training...\n")

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for img, mask in loader:

            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)

            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    print("\nTraining finished!")

    torch.save(model.state_dict(), "unet_model.pth")
    print("U-Net model saved as unet_model.pth")


if __name__ == "__main__":
    main()