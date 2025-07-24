import os
from PIL import Image
import torch
import torchvision.transforms as T
from model import UNetWithSwinEncoder
import numpy as np

# ----------------- 設定 -----------------
MODEL_PATH = "swim_trasnformer_384.pth"
IMG_HEIGHT, IMG_WIDTH = 384, 384
THRESHOLD = 0.5
backbone = 'swin_base_patch4_window12_384'

# 要處理的資料夾清單
input_folders = ["1001_2000", "2001_3000", "3001_4000", "4001_5000", "test"]

# 裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform
transform = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

# 載入模型
model = UNetWithSwinEncoder(backbone_name=backbone).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 處理每個資料夾
for input_root_dir in input_folders:
    output_root_dir = input_root_dir + "_cut"
    os.makedirs(output_root_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_root_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(root, file)
            print(f"Processing: {img_path}")
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"❌ Failed to open {file}: {e}")
                continue

            original_size = image.size

            # 預處理
            input_tensor = transform(image).unsqueeze(0).to(device)

            # 預測
            with torch.no_grad():
                output = model(input_tensor)
                output = torch.sigmoid(output)
                pred = (output > THRESHOLD).float()

            pred_mask = pred.squeeze().cpu().numpy()
            mask_resized = T.Resize(original_size[::-1])(Image.fromarray((pred_mask * 255).astype("uint8")))
            mask_np = np.array(mask_resized) / 255.0

            # 嘗試至少強制裁切，不再跳過
            if np.sum(mask_np) < 5:
                print(f"⚠️ Too little area in mask. Force cropping with minimal region.")
                # 強制設為整張圖的一部分
                x0, y0, x1, y1 = 0, 0, image.width, image.height
            else:
                coords = np.argwhere(mask_np > 0.5)
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)

                pad = 20
                x0 = max(x0 - pad, 0)
                y0 = max(y0 - pad, 0)
                x1 = min(x1 + pad, image.width)
                y1 = min(y1 + pad, image.height)

            # 套用 mask
            mask_np_exp = np.expand_dims(mask_np, axis=-1)
            original_np = np.array(image)
            segmented_np = (original_np * mask_np_exp).astype(np.uint8)
            segmented_img = Image.fromarray(segmented_np)

            # 裁切區域並 resize
            cropped = segmented_img.crop((x0, y0, x1, y1))
            cropped = cropped.convert("RGB")
            cropped.thumbnail((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)

            # 黑底填充
            final_img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0))
            paste_x = (IMG_WIDTH - cropped.width) // 2
            paste_y = (IMG_HEIGHT - cropped.height) // 2
            final_img.paste(cropped, (paste_x, paste_y))

            # 儲存
            rel_path = os.path.relpath(img_path, input_root_dir)
            out_path = os.path.join(output_root_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            final_img.save(out_path)

print("✅ 所有資料夾處理完成！")
