# ---- model.py ----
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)

class UNetWithTimmEncoder(nn.Module):
    def __init__(self, backbone_name='resnet34', out_channels=1):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, features_only=True, pretrained=True)
        enc_channels = self.encoder.feature_info.channels()  # e.g., [64, 128, 256, 512]

        self.up4 = nn.ConvTranspose2d(enc_channels[-1], enc_channels[-1] // 2, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(enc_channels[-1] // 2 + enc_channels[-2], enc_channels[-2])

        self.up3 = nn.ConvTranspose2d(enc_channels[-2], enc_channels[-2] // 2, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(enc_channels[-2] // 2 + enc_channels[-3], enc_channels[-3])

        self.up2 = nn.ConvTranspose2d(enc_channels[-3], enc_channels[-3] // 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(enc_channels[-3] // 2 + enc_channels[-4], enc_channels[-4])

        self.decoder1 = DoubleConv(enc_channels[-4], 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x_input_h, x_input_w = x.shape[2:]  # 記住原圖尺寸
        encs = self.encoder(x)

        x = self.up4(encs[-1])
        if x.shape[2:] != encs[-2].shape[2:]:
            x = F.interpolate(x, size=encs[-2].shape[2:])
        x = torch.cat([x, encs[-2]], dim=1)
        x = self.decoder4(x)

        x = self.up3(x)
        if x.shape[2:] != encs[-3].shape[2:]:
            x = F.interpolate(x, size=encs[-3].shape[2:])
        x = torch.cat([x, encs[-3]], dim=1)
        x = self.decoder3(x)

        x = self.up2(x)
        if x.shape[2:] != encs[-4].shape[2:]:
            x = F.interpolate(x, size=encs[-4].shape[2:])
        x = torch.cat([x, encs[-4]], dim=1)
        x = self.decoder2(x)

        x = self.decoder1(x)
        x = self.final(x)
        x = F.interpolate(x, size=(x_input_h, x_input_w))  # 對齊輸入尺寸
        return x
    
class UNetWithSwinEncoder(nn.Module):
    def __init__(self, backbone_name='swin_base_patch4_window7_224', out_channels=1):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, features_only=True, pretrained=True)
        enc_channels = self.encoder.feature_info.channels()  # 例如 [128, 256, 512, 1024]

        self.up4 = nn.ConvTranspose2d(enc_channels[-1], enc_channels[-1] // 2, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(enc_channels[-1] // 2 + enc_channels[-2], enc_channels[-2])

        self.up3 = nn.ConvTranspose2d(enc_channels[-2], enc_channels[-2] // 2, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(enc_channels[-2] // 2 + enc_channels[-3], enc_channels[-3])

        self.up2 = nn.ConvTranspose2d(enc_channels[-3], enc_channels[-3] // 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(enc_channels[-3] // 2 + enc_channels[-4], enc_channels[-4])

        self.decoder1 = DoubleConv(enc_channels[-4], 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x_input_h, x_input_w = x.shape[2:]  # 保存輸入尺寸
        encs = self.encoder(x)

        encs = [e.permute(0, 3, 1, 2).contiguous() if e.ndim == 4 and e.shape[1] < e.shape[-1] else e for e in encs]

        x = self.up4(encs[-1])
        if x.shape[2:] != encs[-2].shape[2:]:
            x = F.interpolate(x, size=encs[-2].shape[2:])
        x = torch.cat([x, encs[-2]], dim=1)
        x = self.decoder4(x)

        x = self.up3(x)
        if x.shape[2:] != encs[-3].shape[2:]:
            x = F.interpolate(x, size=encs[-3].shape[2:])
        x = torch.cat([x, encs[-3]], dim=1)
        x = self.decoder3(x)

        x = self.up2(x)
        if x.shape[2:] != encs[-4].shape[2:]:
            x = F.interpolate(x, size=encs[-4].shape[2:])
        x = torch.cat([x, encs[-4]], dim=1)
        x = self.decoder2(x)

        x = self.decoder1(x)
        x = self.final(x)

        # 輸出回到原始輸入尺寸（若有需要）
        x = F.interpolate(x, size=(x_input_h, x_input_w))
        return x

