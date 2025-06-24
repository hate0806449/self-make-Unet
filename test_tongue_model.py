import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# ==================== 注意力機制模組 ====================
class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv3 = torch.nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)
        
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
class ImprovedUNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(ImprovedUNet, self).__init__()
        
        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 編碼器
        for feature in features:
            self.encoder.append(self._make_conv_block(in_channels, feature))
            in_channels = feature
        
        # 瓶頸層
        self.bottleneck = self._make_conv_block(features[-1], features[-1] * 2)
        self.attention = AttentionBlock(features[-1] * 2)
        
        # 解碼器
        for feature in reversed(features):
            self.decoder.append(torch.nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._make_conv_block(feature * 2, feature))
        
        # 最終輸出層
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def _make_conv_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
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
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)
        
        return self.sigmoid(self.final_conv(x))

# ==================== 預測器類別 ==================== 
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

# ==================== 主函數 ====================
def main():
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 模型路徑
    model_path = 'best_tongue_model.pth'
    
    # 檢查模型是否存在
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型文件 {model_path}")
        print("請先訓練模型並保存為 best_tongue_model.pth")
        return
    
    # 測試圖片路徑
    test_image_path = 'test.jpg'
    
    # 檢查測試圖片是否存在
    if not os.path.exists(test_image_path):
        print(f"錯誤：找不到測試圖片 {test_image_path}")
        print("請將測試圖片放在同目錄下並命名為 test.jpg")
        return
    
    # 創建預測器
    predictor = TonguePredictor(model_path, device)
    
    # 進行預測並可視化結果
    print("正在預測舌頭分割結果...")
    predictor.visualize_prediction(test_image_path, 'prediction_result.png')
    print("預測完成！結果已保存為 prediction_result.png")

if __name__ == "__main__":
    print("舌頭分割模型測試腳本")
    print("請確保:")
    print("1. 已經訓練過模型並保存為 best_tongue_model.pth")
    print("2. 將測試圖片命名為 test.jpg 並放在同目錄下")
    print("=" * 50)
    main()
