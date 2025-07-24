import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim as optim
from model import UNet, UNetWithTimmEncoder
from dataset import CustomDataset
from train import train_model
from model import UNetWithSwinEncoder

# Config
IMG_HEIGHT, IMG_WIDTH = 384, 384
BATCH_SIZE = 4
NUM_EPOCHS = 20
USE_TIMM = False  # 關掉 TIMM 模型
BACKBONE_NAME = 'swin_base_patch4_window12_384'

IMAGE_DIR = 'images'
MASK_DIR = 'masks'
VAL_IMAGE_DIR = 'image_val'
VAL_MASK_DIR = 'mask_val'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.RandomApply([
    T.ColorJitter(brightness=0.2, contrast=0.1)
        ], p=0.5),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])
val_transform = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

mask_transform = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),  # 這會把 mask 轉成 [1, H, W] 的 tensor 並標準化到 0~1
])


# Dataset and DataLoader
train_dataset = CustomDataset(IMAGE_DIR, MASK_DIR, image_transform=train_transform, mask_transform=mask_transform)
val_dataset = CustomDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, image_transform=val_transform, mask_transform=mask_transform)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
if BACKBONE_NAME.startswith("swin"):
    model = UNetWithSwinEncoder(backbone_name=BACKBONE_NAME).to(device)
elif USE_TIMM:
    model = UNetWithTimmEncoder(BACKBONE_NAME).to(device)
else:
    model = UNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train
train_model(model, train_loader, val_loader, optimizer, device, NUM_EPOCHS, patience=5)
