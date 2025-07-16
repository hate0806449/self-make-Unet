# ---- dataset.py ----
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_filename = self.images[index]
        image_path = os.path.join(self.image_dir, image_filename)

        # 將副檔名改為 .png，取得 mask 檔案
        mask_filename = os.path.splitext(image_filename)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)


        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0).float()  # 確保是二值化的mask

        return image, mask