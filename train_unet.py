import torch
from torch.utils.data import DataLoader, random_split
from models.unet import UNet
from utils.dataset import CamusDataset


# Dice Loss
def dice_loss(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    print("\nLoading dataset...")
    full_dataset = CamusDataset("data/train/images", "data/train/masks")
    print("Total images:", len(full_dataset))

    # Train / Validation split (90:10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"Train: {train_size}, Val: {val_size}")

    print("\nInitializing UNet...")
    model = UNet().to(device)

    bce_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    best_val_loss = float("inf")

    print("\nStarting training...\n")
-
    for epoch in range(epochs):

   
        model.train()
        train_loss = 0

        for img, mask in train_loader:
            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)

           
            loss = bce_loss(pred, mask) + dice_loss(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

       
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device)
                mask = mask.to(device)

                pred = model(img)

                loss = bce_loss(pred, mask) + dice_loss(pred, mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            print(" Best model saved!")

    print("\nTraining finished!")
    print("Best model saved as best_unet_model.pth")


if __name__ == "__main__":
    main()
