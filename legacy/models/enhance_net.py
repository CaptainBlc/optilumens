"""
Egitilebilir pixel iyilestirme agi.
Girdi: RGB goruntu (normalize). Cikti: ayni boyutta iyilestirilmis goruntu.
Hafif mimari: edge/telefon ve gelecekteki dagitim icin uygun.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EnhanceNet(nn.Module):
    """
    U-Net benzeri hafif encoder-decoder; skip connections ile detay korunur.
    Herhangi cozunurlukte calisir (tam konvolusyonel).
    """

    def __init__(self, in_channels=3, out_channels=3, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)       # 32
        self.enc2 = ConvBlock(base, base * 2)         # 64
        self.enc3 = ConvBlock(base * 2, base * 4)     # 128
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.out(d1))
