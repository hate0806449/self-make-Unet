import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader ,Dataset
from PIL import Image
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,features=[64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        # -1因為他還會變成1024，但我們要512
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
    
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature *2,feature,2,2))
            self.ups.append(DoubleConv(feature * 2, feature))
        self.final_conv=nn.Conv2d(features[0], out_channels, 1)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, (2, 2))
        x = self.bottleneck(x)
        skip_connections.reverse()
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]
            concat = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat)
        return self.final_conv(x)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
model= UNet()
model = model.cuda()

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path).convert('L'))

        image = self.transform(image)
        mask = self.transform(mask)

        mask = (mask > 0).float()  # 二值化 & 轉 float
        return image, mask
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

BATCH_SIZE = 16
NUM_EPOCHS = 20
IMG_WIDTH = 240
IMG_HEIGHT = 160

all_data= CustomDataset('images', 'masks', T.Compose([
    T.ToTensor(),
    T.Resize((IMG_WIDTH, IMG_HEIGHT))]))
train_data, val_data = torch.utils.data.random_split(all_data,[0.7, 0.3])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

def train(model, num_epochs, train_loader, optimizer,  print_every=30):
    for epoch in range(num_epochs):
        for count,(x,y) in enumerate(train_loader):
            model.train()
            x=x.to(device)
            y=y.to(device)
            out=model(x)
            if count%print_every == 0:
                eval(model, val_loader, epoch)
            out=torch.sigmoid(out)
            loss=loss_function(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
def eval (model, val_loader, epoch):
    model.eval()
    num_correct = 0
    num_pixels = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            out_img= model(x)
            probability=torch.sigmoid(out_img)
            predictions=probability > 0.5
            num_correct+=(predictions==y).sum()
            num_pixels+=BATCH_SIZE * IMG_WIDTH * IMG_HEIGHT
        print(f'Epoch {epoch+1}, Acc:{num_correct/num_pixels}')
train(model, NUM_EPOCHS,train_loader,optimizer)
