# ---- train.py ----
import torch
import torch.nn as nn
from utils import dice_score

def eval_model(model, val_loader, device, epoch):
    model.eval()
    total_dice = 0
    total_loss = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = torch.sigmoid(model(x))
            loss = criterion(out, y)
            preds = (out > 0.5).float()
            total_dice += dice_score(preds, y).item()
            total_loss += loss.item()

    avg_dice = total_dice / len(val_loader)
    avg_loss = total_loss / len(val_loader)
    print(f"[Validation] Epoch {epoch+1} | Dice: {avg_dice:.4f} | Loss: {avg_loss:.4f}")
    return avg_dice


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=20, patience=5):
    criterion = nn.BCELoss()
    best_dice = 0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = torch.sigmoid(model(x))
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1} | Loss: {avg_train_loss:.4f}")

        val_dice = eval_model(model, val_loader, device, epoch)

        if val_dice > best_dice:
            best_dice = val_dice
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"[Model Saved] New best model at epoch {epoch+1} with Dice {val_dice:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"[Early Stop] No improvement in Dice for {patience} epochs. Stopping training.")
                break