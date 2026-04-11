"""
StyleGAN2 Clean Architecture — No custom CUDA ops.

Reproduced from BasicSR / GFPGAN source code (MIT License).
https://github.com/TencentARC/GFPGAN
https://github.com/xinntao/BasicSR

Used as the backbone generator inside GFPGANv1Clean.
All operations are standard PyTorch — works on any platform.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormStyleCode(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale,
                        self.bias * self.lr_mul if self.bias is not None else None)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat,
                 demodulate=True, sample_mode=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.padding = kernel_size // 2

        self.modulation = EqualLinear(num_style_feat, in_channels, bias_init=1)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
            / math.sqrt(in_channels * kernel_size ** 2))

    def forward(self, x, style):
        b, c, h, w = x.shape
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            dcoefs = (weight.pow(2).sum([2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * dcoefs.view(b, self.out_channels, 1, 1, 1)

        if self.sample_mode == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                              align_corners=False)
        elif self.sample_mode == 'downsample':
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear',
                              align_corners=False)

        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        return out.view(b, self.out_channels, *out.shape[2:])


class StyleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat,
                 demodulate=True, sample_mode=None):
        super().__init__()
        self.modulated_conv = ModulatedConv2d(in_channels, out_channels, kernel_size,
                                              num_style_feat, demodulate, sample_mode)
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, style, noise=None):
        out = self.modulated_conv(x, style)
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise + self.bias
        return self.activate(out)


class ToRGB(nn.Module):
    def __init__(self, in_channels, num_style_feat, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(in_channels, 3, 1, num_style_feat,
                                              demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.modulated_conv(x, style) + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='bilinear',
                                     align_corners=False)
            out = out + skip
        return out


class ConstantInput(nn.Module):
    def __init__(self, num_channel, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        return self.weight.repeat(batch, 1, 1, 1)


class StyleGAN2GeneratorCSFT(nn.Module):
    """StyleGAN2 generator with Condition SFT (GFPGAN backbone)."""

    def __init__(self, out_size, num_style_feat=512, num_mlp=8,
                 channel_multiplier=2, lr_mlp=0.01,
                 narrow=1, sft_half=False):
        super().__init__()
        self.num_style_feat = num_style_feat
        self.sft_half = sft_half

        # Mapping network
        mapping_layers = [NormStyleCode()]
        for _ in range(num_mlp):
            mapping_layers.append(
                EqualLinear(num_style_feat, num_style_feat,
                            lr_mul=lr_mlp, bias_init=0))
            mapping_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.style_mlp = nn.Sequential(*mapping_layers)

        # Channel sizes
        channels = {
            '4':  int(512 * narrow),
            '8':  int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow),
        }
        self.channels = channels

        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(channels['4'], channels['4'], 3, num_style_feat)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, upsample=False)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latents = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channels = channels['4']
        for i in range(3, self.log_size + 1):
            out_ch = channels[str(2 ** i)]
            self.style_convs.append(
                StyleConv(in_channels, out_ch, 3, num_style_feat,
                          sample_mode='upsample'))
            self.style_convs.append(
                StyleConv(out_ch, out_ch, 3, num_style_feat))
            self.to_rgbs.append(ToRGB(out_ch, num_style_feat))
            in_channels = out_ch

    def make_noise(self):
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def forward(self, styles, conditions, input_is_latent=False,
                noise=None, randomize_noise=True,
                truncation=1, truncation_latent=None,
                inject_index=None, return_latents=False):

        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]

        if truncation < 1:
            style_t = []
            for s in styles:
                style_t.append(truncation_latent + truncation * (s - truncation_latent))
            styles = style_t

        # Replicate or mix styles
        if len(styles) == 1:
            latent = styles[0].unsqueeze(1).repeat(1, self.num_latents, 1)
        elif len(styles) == 2:
            inject_index = inject_index or random.randint(1, self.num_latents - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latents - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        else:
            latent = torch.stack(styles, dim=1)

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self, f'noise_{i}', None)
                         for i in range(self.num_layers)]

        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        condition_idx = 0
        for conv1, conv2, to_rgb in zip(
                self.style_convs[::2],
                self.style_convs[1::2],
                self.to_rgbs):

            out = conv1(out, latent[:, i], noise=noise[i])

            # SFT condition modulation
            if conditions and condition_idx < len(conditions):
                cond = conditions[condition_idx]
                if self.sft_half:
                    out_s, out_l = out.chunk(2, dim=1)
                    if cond.shape[2:] == out_s.shape[2:]:
                        out_s = out_s * cond[:, :out_s.shape[1]] + cond[:, out_s.shape[1]:]
                    out = torch.cat([out_s, out_l], dim=1)
                elif cond.shape[2:] == out.shape[2:]:
                    out = out * cond[:, :out.shape[1]] + cond[:, out.shape[1]:]

            out = conv2(out, latent[:, i + 1], noise=noise[i + 1])
            condition_idx += 1

            if conditions and condition_idx < len(conditions):
                cond = conditions[condition_idx]
                if self.sft_half:
                    out_s, out_l = out.chunk(2, dim=1)
                    if cond.shape[2:] == out_s.shape[2:]:
                        out_s = out_s * cond[:, :out_s.shape[1]] + cond[:, out_s.shape[1]:]
                    out = torch.cat([out_s, out_l], dim=1)
                elif cond.shape[2:] == out.shape[2:]:
                    out = out * cond[:, :out.shape[1]] + cond[:, out.shape[1]:]
            condition_idx += 1

            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip
        return (image, latent) if return_latents else (image, None)
