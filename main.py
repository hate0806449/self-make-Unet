import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ==================== 資料集類別 ====================
class TongueDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        
        # 獲取所有圖片文件名
        self.images = []
        for file in os.listdir(image_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
                self.images.append(file)
        
        # 默認的數據增強
        if transform is None:
            if is_train:
                self.transform = A.Compose([
                    A.Resize(512, 512),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.RandomGamma(p=0.2),
                    A.ElasticTransform(p=0.2),
                    A.GridDistortion(p=0.2),
                    A.OpticalDistortion(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 尋找對應的mask文件
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 讀取圖片和mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # 將mask二值化
            mask = (mask > 127).astype(np.uint8)
        else:
            # 如果沒有mask，創建空的mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 應用變換
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.float()

# ==================== 注意力機制模組 ====================
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 計算注意力權重
        proj_query = self.conv1(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.conv2(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.conv3(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        out = self.gamma * out + x
        return out

# ==================== 改進的U-Net模型 ====================
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(ImprovedUNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 編碼器
        for feature in features:
            self.encoder.append(self._make_conv_block(in_channels, feature))
            in_channels = feature
        
        # 瓶頸層
        self.bottleneck = self._make_conv_block(features[-1], features[-1] * 2)
        self.attention = AttentionBlock(features[-1] * 2)
        
        # 解碼器
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._make_conv_block(feature * 2, feature))
        
        # 最終輸出層
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        skip_connections = []
        
        # 編碼器路徑
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶頸層
        x = self.bottleneck(x)
        x = self.attention(x)
        
        skip_connections = skip_connections[::-1]
        
        # 解碼器路徑
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)
        
        return self.sigmoid(self.final_conv(x))

# ==================== 損失函數 ====================
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

# ==================== 評估指標 ====================
def calculate_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0
    return (intersection / union).item()

def calculate_dice(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2 * intersection) / (pred_binary.sum() + target_binary.sum())
    
    return dice.item()

# ==================== 訓練器類別 ====================
class TongueSegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # 修復：移除 verbose 參數
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.val_dices = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for images, masks in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 確保輸出和目標的維度匹配
            if outputs.dim() == 4 and masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                
                if outputs.dim() == 4 and masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # 計算IoU和Dice分數
                for i in range(outputs.shape[0]):
                    iou = calculate_iou(outputs[i], masks[i])
                    dice = calculate_dice(outputs[i], masks[i])
                    total_iou += iou
                    total_dice += dice
        
        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / (len(self.val_loader) * self.val_loader.batch_size)
        avg_dice = total_dice / (len(self.val_loader) * self.val_loader.batch_size)
        
        return avg_loss, avg_iou, avg_dice
    
    def train(self, num_epochs, save_path='best_tongue_model.pth'):
        best_iou = 0
        
        for epoch in range(num_epochs):
            # 訓練
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 驗證
            val_loss, val_iou, val_dice = self.validate()
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            self.val_dices.append(val_dice)
            
            # 學習率調整
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # 手動打印學習率變化
            if new_lr != current_lr:
                print(f'Learning rate reduced from {current_lr:.2e} to {new_lr:.2e}')
            
            # 保存最佳模型
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_iou': best_iou,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_ious': self.val_ious,
                    'val_dices': self.val_dices
                }, save_path)
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}')
            print(f'Best IoU: {best_iou:.4f}')
            print('-' * 60)
    
    def plot_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 損失曲線
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # IoU曲線
        axes[0, 1].plot(self.val_ious, label='Val IoU', color='green')
        axes[0, 1].set_title('Validation IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        
        # Dice分數曲線
        axes[1, 0].plot(self.val_dices, label='Val Dice', color='red')
        axes[1, 0].set_title('Validation Dice Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        
        # 清空最後一個子圖
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

# ==================== 預測和可視化 ====================
class TonguePredictor:
    def __init__(self, model_path, device):
        self.device = device
        self.model = ImprovedUNet()
        
        # 載入模型
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # 預處理轉換
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def predict_single_image(self, image_path, threshold=0.5):
        # 讀取和預處理圖片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # 應用轉換
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.squeeze().cpu().numpy()
        
        # 二值化預測結果
        binary_mask = (prediction > threshold).astype(np.uint8) * 255
        
        # 調整回原始大小
        binary_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]))
        
        return binary_mask, prediction
    
    def visualize_prediction(self, image_path, save_path=None):
        # 讀取原始圖片
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # 進行預測
        mask, _ = self.predict_single_image(image_path)
        
        # 創建可視化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始圖片
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 預測mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        # 疊加結果
        overlay = original.copy()
        mask_colored = np.zeros_like(original)
        mask_colored[:, :, 0] = mask  # 紅色通道
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Result')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# ==================== 主要使用範例 ====================
def main():
    # 設定參數
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 資料路徑 - 請根據您的實際資料夾結構修改
    base_dir = r'C:\Users\user\Desktop\tongue'  # 修改為您的主資料夾路徑
    train_image_dir = os.path.join(base_dir, 'dataset', 'train', 'images')
    train_mask_dir = os.path.join(base_dir, 'dataset', 'train', 'masks')
    val_image_dir = os.path.join(base_dir, 'dataset', 'val', 'images')
    val_mask_dir = os.path.join(base_dir, 'dataset', 'val', 'masks')
    
    # 檢查資料夾是否存在
    required_dirs = [train_image_dir, train_mask_dir, val_image_dir, val_mask_dir]
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("錯誤：找不到以下資料夾，請先創建並放入資料：")
        for missing_dir in missing_dirs:
            print(f"  - {missing_dir}")
        print("\n建議的資料夾結構：")
        print(f"{base_dir}/")
        print("├── dataset/")
        print("│   ├── train/")
        print("│   │   ├── images/  (放入訓練用的舌頭圖片)")
        print("│   │   └── masks/   (放入對應的分割遮罩)")
        print("│   └── val/")
        print("│       ├── images/  (放入驗證用的舌頭圖片)")
        print("│       └── masks/   (放入對應的分割遮罩)")
        print("\n請創建資料夾並放入資料後再執行程式。")
        return
    
    # 檢查資料夾中是否有檔案
    train_images_count = len([f for f in os.listdir(train_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg','.bmp'))])
    val_images_count = len([f for f in os.listdir(val_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg','.bmp'))])
    
    if train_images_count == 0:
        print(f"錯誤：訓練圖片資料夾是空的: {train_image_dir}")
        return
    
    if val_images_count == 0:
        print(f"錯誤：驗證圖片資料夾是空的: {val_image_dir}")
        return
    
    print(f"找到 {train_images_count} 張訓練圖片")
    print(f"找到 {val_images_count} 張驗證圖片")
    
    # 創建資料集和資料載入器
    try:
        train_dataset = TongueDataset(train_image_dir, train_mask_dir, is_train=True)
        val_dataset = TongueDataset(val_image_dir, val_mask_dir, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
        
        print(f"資料載入成功！")
        print(f"訓練資料集大小: {len(train_dataset)}")
        print(f"驗證資料集大小: {len(val_dataset)}")
        
    except Exception as e:
        print(f"載入資料時發生錯誤: {e}")
        return
    
    # 創建模型
    model = ImprovedUNet(in_channels=3, out_channels=1)
    
    # 創建訓練器
    trainer = TongueSegmentationTrainer(model, train_loader, val_loader, device)
    
    # 開始訓練
    print("開始訓練舌頭分割模型...")
    trainer.train(num_epochs=9, save_path='best_tongue_model.pth')
    
    # 繪製訓練歷史
    trainer.plot_training_history()
    
    # 測試預測 (需要修改為實際的測試圖片路徑)
    # predictor = TonguePredictor('best_tongue_model.pth', device)
    # test_image_path = 'path/to/test/image.jpg'
    # predictor.visualize_prediction(test_image_path, 'prediction_result.png')
    
    print("訓練完成！")

if __name__ == "__main__":
    # 注意：執行前請先安裝所需套件
    print("請先安裝以下套件：")
    print("pip install torch torchvision opencv-python pillow matplotlib scikit-learn albumentations segmentation-models-pytorch")
    
    main()