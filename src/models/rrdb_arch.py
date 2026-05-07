"""
RRDBNet — Real-ESRGAN backbone (Residual-in-Residual Dense Block).

Pure PyTorch implementation that loads the official Real-ESRGAN weights
without depending on `basicsr` (same strategy used for GFPGAN in this
project — basicsr is incompatible with Python 3.13).

Reference paper:
    Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
    with Pure Synthetic Data", ICCV Workshops 2021.

State-dict layout (matches the official RealESRGAN_x2plus.pth and x4plus.pth):
    conv_first.weight / bias
    body.{0..22}.rdb{1,2,3}.conv{1..5}.weight / bias
    conv_body.weight / bias
    conv_up1.weight / bias
    conv_up2.weight / bias
    conv_hr.weight / bias
    conv_last.weight / bias

Scale notes:
    - x4 model: spatial input == native; output = input × 4
    - x2 model: input is pixel-unshuffled by 2 first (effectively /2),
                then upsampled twice → net output = input × 2
    - x1 (rare): pixel-unshuffle by 4 → net output = input × 1
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Pixel unshuffle (inverse of PixelShuffle) ───────────────────────

def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Inverse of nn.PixelShuffle. Reduces spatial resolution by `scale` and
    expands channels by `scale**2`. Used by the x2 / x1 RRDBNet variants
    so the body still receives the same channel count as the x4 path.

    Available natively in PyTorch ≥ 1.8 as F.pixel_unshuffle, used here
    explicitly for clarity and version safety.
    """
    b, c, h, w = x.size()
    assert h % scale == 0 and w % scale == 0, \
        "Input H/W must be divisible by scale {}".format(scale)
    x = x.view(b, c, h // scale, scale, w // scale, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * (scale ** 2), h // scale, w // scale)


# ── Residual Dense Block ────────────────────────────────────────────

class ResidualDenseBlock(nn.Module):
    """
    Five 3×3 convs with dense skip connections + residual scaling 0.2.

    Channel growth pattern:
        conv1: nf            → ng
        conv2: nf+ng         → ng
        conv3: nf+2ng        → ng
        conv4: nf+3ng        → ng
        conv5: nf+4ng        → nf      (back to input width)
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        nf, ng = num_feat, num_grow_ch
        self.conv1 = nn.Conv2d(nf,         ng, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + 1 * ng, ng, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * ng, ng, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * ng, ng, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * ng, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


# ── Residual-in-Residual Dense Block (RRDB) ─────────────────────────

class RRDB(nn.Module):
    """Three stacked ResidualDenseBlocks with outer residual scaling 0.2."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


# ── Full RRDBNet ────────────────────────────────────────────────────

class RRDBNet(nn.Module):
    """
    Real-ESRGAN backbone. Stack: pixel-unshuffle (if scale<4) → conv_first
    → 23 RRDBs → conv_body (residual) → 2× nearest upsample × 2 layers
    → conv_hr → conv_last → image.

    Parameters
    ----------
    num_in_ch     int  Input channels (3 = RGB).
    num_out_ch    int  Output channels (3 = RGB).
    scale         int  Net super-resolution factor: 4, 2, or 1.
    num_feat      int  Width (Real-ESRGAN: 64).
    num_block     int  RRDB count (Real-ESRGAN: 23).
    num_grow_ch   int  Dense growth channels (Real-ESRGAN: 32).
    """

    def __init__(
        self,
        num_in_ch:   int = 3,
        num_out_ch:  int = 3,
        scale:       int = 4,
        num_feat:    int = 64,
        num_block:   int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale

        # x2 / x1 variants pixel-unshuffle the input → channel multiplier
        if scale == 2:
            in_ch = num_in_ch * 4
        elif scale == 1:
            in_ch = num_in_ch * 16
        else:
            in_ch = num_in_ch  # x4 path

        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[
            RRDB(num_feat, num_grow_ch) for _ in range(num_block)
        ])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling stack — always two ×2 layers (×4 net at this point)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Pre-shuffle so the body sees consistent channel width
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        # 2) Trunk
        feat      = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat      = feat + body_feat

        # 3) Upsample ×2 ×2 → ×4 spatially from `feat`
        feat = self.lrelu(self.conv_up1(
            F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(
            F.interpolate(feat, scale_factor=2, mode="nearest")))

        # 4) Project back to RGB
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ── Convenience factory ─────────────────────────────────────────────

def realesrgan_x2plus() -> RRDBNet:
    """Architecture matching RealESRGAN_x2plus.pth (default for general use)."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, scale=2,
                   num_feat=64, num_block=23, num_grow_ch=32)


def realesrgan_x4plus() -> RRDBNet:
    """Architecture matching RealESRGAN_x4plus.pth (max-quality, slower)."""
    return RRDBNet(num_in_ch=3, num_out_ch=3, scale=4,
                   num_feat=64, num_block=23, num_grow_ch=32)
