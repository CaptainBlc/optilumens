"""
GFPGANv1Clean Architecture

Reproduced from GFPGAN (MIT License).
https://github.com/TencentARC/GFPGAN

Differences from original:
  - No basicsr / custom CUDA dependency
  - Pure PyTorch, Python 3.13 compatible
  - Weight-compatible with official GFPGANv1.3.pth
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stylegan2_clean import StyleGAN2GeneratorCSFT


class ResBlock(nn.Module):
    """
    Residual block for condition extraction (matches original GFPGAN).

    Two key behaviours we must reproduce to load `GFPGANv1.3.pth`:
      1. conv1 keeps the channel count identical to the input width;
         conv2 is the layer that performs the channel transition.
         (The opposite ordering causes shape-mismatch on load.)
      2. The block also rescales spatial resolution: encoder blocks
         downsample by ½, decoder blocks upsample by 2. Without this,
         the encoder feature map never reaches 4×4, and the linear
         projection blows up to ~67 M elements.
    """

    def __init__(self, in_channels, out_channels, mode: str = "down"):
        super().__init__()
        assert mode in ("down", "up", "none"), \
            "mode must be 'down', 'up', or 'none'"
        self.scale_factor = {"down": 0.5, "up": 2.0, "none": 1.0}[mode]

        self.conv1 = nn.Conv2d(in_channels, in_channels,  3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip  = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        if self.scale_factor != 1.0:
            out = F.interpolate(out, scale_factor=self.scale_factor,
                                mode="bilinear", align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)

        x_skip = x
        if self.scale_factor != 1.0:
            x_skip = F.interpolate(x_skip, scale_factor=self.scale_factor,
                                   mode="bilinear", align_corners=False)
        return out + self.skip(x_skip)


class GFPGANv1Clean(nn.Module):
    """
    GFPGAN v1.3 — Generative Facial Prior GAN (Clean Version).

    Architecture overview:
        1. U-Net encoder extracts degradation features from input face
        2. Features are projected into SFT conditions
        3. StyleGAN2 generator synthesizes the restored face
           guided by spatial feature transform (SFT) conditions
        4. Decoder reconstructs final image

    Args:
        out_size (int):           Output resolution (512 for GFPGANv1.3)
        num_style_feat (int):     Style feature dim (512)
        channel_multiplier (int): StyleGAN2 channel multiplier (2)
        decoder_load_path (str):  Unused (kept for weight compatibility)
        fix_decoder (bool):       Unused
        num_mlp (int):            Mapping network depth (8)
        input_is_latent (bool):   Whether input is already in W space
        different_w (bool):       Per-layer style injection
        narrow (float):           Channel narrowing factor (1)
        sft_half (bool):          Apply SFT to half channels only (True)
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 channel_multiplier=2,
                 decoder_load_path=None,
                 fix_decoder=True,
                 num_mlp=8,
                 input_is_latent=False,
                 different_w=False,
                 narrow=1,
                 sft_half=False):
        super().__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5
        channels = {
            '4':  int(512 * unet_narrow),
            '8':  int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow),
        }

        self.log_size = int(math.log(out_size, 2))

        # ── Encoder (U-Net) ─────────────────────────────────────
        first_out_size = 2 ** (int(math.log(out_size, 2)))
        self.conv_body_first = nn.Conv2d(3, channels[str(first_out_size)], 1)

        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            self.conv_body_down.append(
                ResBlock(channels[str(2 ** i)], channels[str(2 ** (i - 1))],
                         mode="down"))

        self.final_conv = nn.Conv2d(channels['4'], channels['4'], 3, 1, 1)

        # ── Style code from encoder features ────────────────────
        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat
        self.final_linear = nn.Linear(channels['4'] * 4 * 4, linear_out_channel)

        # ── Decoder (U-Net) ─────────────────────────────────────
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.conv_body_up.append(
                ResBlock(channels[str(2 ** (i - 1))], channels[str(2 ** i)],
                         mode="up"))

        # ── SFT condition layers ─────────────────────────────────
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[str(2 ** i)]
            sft_out = out_channels if not sft_half else out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out, 3, 1, 1)))
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out, 3, 1, 1)))

        # ── StyleGAN2 generator ──────────────────────────────────
        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half)

        # ── Tonemapping / output conv ────────────────────────────
        self.toRGB = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(3, 3, 3, 1, 1))

    def forward(self, x, return_latents=False, return_rgb=True,
                randomize_noise=True, **kwargs):
        conditions = []
        unet_skips = []

        # ── Encoder pass ────────────────────────────────────────
        enc = self.conv_body_first(x)
        for i in range(self.log_size - 2):
            enc = self.conv_body_down[i](enc)
            unet_skips.insert(0, enc)

        enc = F.leaky_relu_(self.final_conv(enc), negative_slope=0.2)

        # ── Style code ───────────────────────────────────────────
        style_code = self.final_linear(enc.view(enc.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1,
                                         self.num_style_feat)

        # ── Decoder + SFT conditions ─────────────────────────────
        # ResBlock(mode="up") already upsamples, so no extra interpolate.
        dec = None
        for i in range(self.log_size - 2):
            enc_skip = unet_skips[i]
            in_feat = enc_skip if dec is None else (enc_skip + dec)
            dec = self.conv_body_up[i](in_feat)

            scale = self.condition_scale[i](dec)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](dec)
            conditions.append(shift.clone())

        # ── StyleGAN2 synthesis ──────────────────────────────────
        if self.different_w:
            styles = [style_code]
        else:
            styles = [style_code] * 2

        image, latent = self.stylegan_decoder(
            styles,
            conditions,
            return_latents=return_latents,
            input_is_latent=self.input_is_latent,
            randomize_noise=randomize_noise)

        image = self.toRGB(image) if return_rgb else image
        return image, latent
