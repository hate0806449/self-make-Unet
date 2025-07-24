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
        self.enc_channels = self.encoder.feature_info.channels()  # e.g., [64, 128, 256, 512]

        # Bottom feature map
        self.bottleneck = DoubleConv(self.enc_channels[-1], self.enc_channels[-1])

        # Decoder blocks
        self.up4 = nn.ConvTranspose2d(self.enc_channels[-1], self.enc_channels[-2], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(self.enc_channels[-2] * 2, self.enc_channels[-2])

        self.up3 = nn.ConvTranspose2d(self.enc_channels[-2], self.enc_channels[-3], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(self.enc_channels[-3] * 2, self.enc_channels[-3])

        self.up2 = nn.ConvTranspose2d(self.enc_channels[-3], self.enc_channels[-4], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(self.enc_channels[-4] * 2, self.enc_channels[-4])

        # One more up to 64 channels (shallow decoder)
        self.up1 = nn.ConvTranspose2d(self.enc_channels[-4], 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + self.enc_channels[0], 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        input_h, input_w = x.shape[2:]
        feats = self.encoder(x)

        x = self.bottleneck(feats[-1])

        x = self.up4(x)
        if x.shape[2:] != feats[-2].shape[2:]:
            x = F.interpolate(x, size=feats[-2].shape[2:])
        x = torch.cat([x, feats[-2]], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        if x.shape[2:] != feats[-3].shape[2:]:
            x = F.interpolate(x, size=feats[-3].shape[2:])
        x = torch.cat([x, feats[-3]], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        if x.shape[2:] != feats[-4].shape[2:]:
            x = F.interpolate(x, size=feats[-4].shape[2:])
        x = torch.cat([x, feats[-4]], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        if x.shape[2:] != feats[0].shape[2:]:
            x = F.interpolate(x, size=feats[0].shape[2:])
        x = torch.cat([x, feats[0]], dim=1)
        x = self.dec1(x)

        x = self.final(x)
        x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=False)
        return x

    
class UNetWithSwinEncoder(nn.Module):
    def __init__(self, backbone_name='swin_base_patch4_window12_384', out_channels=1):
        super().__init__()
        # Swin Transformer 作為 Encoder
        self.encoder = timm.create_model(backbone_name, features_only=True, pretrained=True)
        enc_channels = self.encoder.feature_info.channels()  # [128, 256, 512, 1024]

        # Decoder 結構
        self.up4 = nn.ConvTranspose2d(enc_channels[-1], enc_channels[-1] // 2, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(enc_channels[-1] // 2 + enc_channels[-2], enc_channels[-2])

        self.up3 = nn.ConvTranspose2d(enc_channels[-2], enc_channels[-2] // 2, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(enc_channels[-2] // 2 + enc_channels[-3], enc_channels[-3])

        self.up2 = nn.ConvTranspose2d(enc_channels[-3], enc_channels[-3] // 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(enc_channels[-3] // 2 + enc_channels[-4], enc_channels[-4])

        self.up1 = nn.ConvTranspose2d(enc_channels[-4], enc_channels[-4] // 2, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(enc_channels[-4] // 2, 64)

        # 最終輸出層
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x_input_h, x_input_w = x.shape[2:]  # 原始輸入尺寸
        encs = self.encoder(x)  # Swin 輸出多層特徵圖 (B, C, H, W)

        # 若特徵圖通道不在 dim=1，就轉成 (B, C, H, W)
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

        x = self.up1(x)
        x = self.decoder1(x)

        x = self.final(x)
        x = F.interpolate(x, size=(x_input_h, x_input_w), mode='bilinear', align_corners=False)  # 回原圖大小
        return x
class UNetWithMobileNetEncoder(nn.Module):
    def __init__(self, backbone_name='mobilenetv2_100', out_channels=1):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, features_only=True, pretrained=True)
        self.enc_channels = self.encoder.feature_info.channels()  # e.g. [24, 32, 96, 1280]

        # Bottleneck
        self.bottleneck = DoubleConv(self.enc_channels[-1], self.enc_channels[-1])

        # Decoder blocks
        self.up4 = nn.ConvTranspose2d(self.enc_channels[-1], self.enc_channels[-2], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(self.enc_channels[-2] + self.enc_channels[-2], self.enc_channels[-2])

        self.up3 = nn.ConvTranspose2d(self.enc_channels[-2], self.enc_channels[-3], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(self.enc_channels[-3] + self.enc_channels[-3], self.enc_channels[-3])

        self.up2 = nn.ConvTranspose2d(self.enc_channels[-3], self.enc_channels[-4], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(self.enc_channels[-4] + self.enc_channels[-4], self.enc_channels[-4])

        self.up1 = nn.ConvTranspose2d(self.enc_channels[-4], 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        input_h, input_w = x.shape[2:]
        feats = self.encoder(x)

        x = self.bottleneck(feats[-1])

        x = self.up4(x)
        if x.shape[2:] != feats[-2].shape[2:]:
            x = F.interpolate(x, size=feats[-2].shape[2:])
        x = torch.cat([x, feats[-2]], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        if x.shape[2:] != feats[-3].shape[2:]:
            x = F.interpolate(x, size=feats[-3].shape[2:])
        x = torch.cat([x, feats[-3]], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        if x.shape[2:] != feats[-4].shape[2:]:
            x = F.interpolate(x, size=feats[-4].shape[2:])
        x = torch.cat([x, feats[-4]], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        x = self.final(x)
        x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=False)
        return x

