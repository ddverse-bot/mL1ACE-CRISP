import torch
from torch.utils.data import DataLoader, random_split
from models.unet import UNet
from utils.dataset import CamusDataset
import wandb



def dice_loss(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)



def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)



def main():

    
    wandb.init(
        project="unet-segmentation",
        name="baseline-dice-bce",
        config={
            "epochs": 100,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "loss": "BCE + Dice",
            "optimizer": "Adam",
            "dataset": "CAMUS"
        }
    )

    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    print("\nLoading dataset...")
    full_dataset = CamusDataset("data/train/images", "data/train/masks")
    print("Total images:", len(full_dataset))

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"Train: {train_size}, Val: {val_size}")

    
    print("\nInitializing UNet...")
    model = UNet().to(device)

    bce_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    epochs = config.epochs
    best_val_loss = float("inf")

    print("\nStarting training...\n")

    
    for epoch in range(epochs):

        
        model.train()
        train_loss = 0

        for img, mask in train_loader:
            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)
            pred = torch.sigmoid(pred)

            loss = bce_loss(pred, mask) + dice_loss(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        
        model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device)
                mask = mask.to(device)

                pred = model(img)
                pred = torch.sigmoid(pred)

                loss = bce_loss(pred, mask) + dice_loss(pred, mask)

                val_loss += loss.item()
                val_dice += dice_score(pred, mask).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f}"
        )

        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice
        })

        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            print(" Best model saved!")

            # log model artifact
            wandb.save("best_unet_model.pth")

    print("\nTraining finished!")
    print("Best model saved as best_unet_model.pth")

    wandb.finish()



if __name__ == "__main__":
    main()
