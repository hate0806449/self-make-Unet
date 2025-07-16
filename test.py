# ---- test.py ----
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import ImageEnhance
import torch
import torchvision.transforms as T
from PIL import Image
from model import UNet, UNetWithTimmEncoder  # 你的模型定義檔
import matplotlib.pyplot as plt

# 參數設定
MODEL_PATH = "best_model.pth"
IMAGE_PATH = "me.jpg"
OUTPUT_PATH = "pred_test.png"  # 直接存在執行資料夾
IMG_HEIGHT, IMG_WIDTH = 400, 400
THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform（確保和訓練時一致）
transform = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),  # 若你訓練時沒有 Normalization，可移除這行
])

# 載入模型
model = UNetWithTimmEncoder(backbone_name='resnet34').to(device)
#model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 載入並預處理圖片
image = Image.open(IMAGE_PATH).convert("RGB")
#enhancer = ImageEnhance.Brightness(image)  # 建立亮度增強器 
#image = enhancer.enhance(1.5)  # 增強亮度
input_tensor = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]

# 推論
with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output)
    pred = (output > THRESHOLD).float()

# 準備可視化圖像
# 將輸入圖轉為 numpy 並反標準化以便顯示
input_image = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()  # [H, W, C]
input_image = (input_image * 0.5 + 0.5).clip(0, 1)  # 如果使用 Normalize([0.5], [0.5])

# 機率圖（0~1）
prob_map = output.squeeze().cpu().numpy()

# 二值化後的圖像 (0 or 255)
pred_img = (pred.squeeze().cpu().numpy() * 255).astype("uint8")

# 顯示
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(input_image)
plt.title("Input (400x400)")

plt.subplot(1, 3, 2)
plt.imshow(prob_map, cmap='jet')
plt.title("Probability Map")

plt.subplot(1, 3, 3)
plt.imshow(pred_img, cmap='gray')
plt.title(f"Threshold > {THRESHOLD}")

plt.tight_layout()
plt.show()

# 儲存結果
Image.fromarray(pred_img).save(OUTPUT_PATH)
print(f"Segmentation mask saved to {OUTPUT_PATH}")
