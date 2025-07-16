# ---- utils.py ----
def compute_accuracy(preds, labels):
    return (preds == labels).sum().item() / torch.numel(preds)

def dice_score(preds, targets, epsilon=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + epsilon) / (preds.sum() + targets.sum() + epsilon)