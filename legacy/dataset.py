"""
Egitim verisi icin iki senaryo:

1) Eslestirilmis ciftler (paired):
   - Klasor yapisi: data/train/input/ ve data/train/target/
   - Ayni isimde dosyalar (001.png hem input hem target'ta).

2) Eslestirilmemis / tek klasor (unpaired / self-supervised):
   - Klasor: data/train/images/ (veya herhangi bir klasor)
   - Her goruntu kendisinin hedefi kabul edilir; input icin rastgele
     bozulmus (dusuk isik, blur, noise) versiyon olusturulur.
   - Boylece elinizde sadece karisik fotograflar olsa bile ag ogrenebilir.
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


class PairedImageDataset(Dataset):
    """
    Gercek eslenmis ciftler varsa kullanilir (low/high).
    """

    def __init__(self, input_dir, target_dir, size=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.size = size  # (H, W) or None = orijinal
        self.files = []
        if os.path.isdir(input_dir) and os.path.isdir(target_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith(IMG_EXTS):
                    if os.path.isfile(os.path.join(target_dir, f)):
                        self.files.append(f)

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if self.size:
            img = cv2.resize(img, (self.size[1], self.size[0]))
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        name = self.files[idx]
        x = self._load(os.path.join(self.input_dir, name))
        y = self._load(os.path.join(self.target_dir, name))
        if x is None or y is None:
            return self.__getitem__((idx + 1) % len(self.files))
        return x, y


class SelfSupervisedImageDataset(Dataset):
    """
    Eslestirilmis cift yoksa kullanilir.
    Tek klasorden goruntu okur, input icin bozulmus versiyon uretir.
    Target = orijinal temiz goruntu kabul edilir.
    """

    def __init__(self, image_dir, size=None):
        self.image_dir = image_dir
        self.size = size
        self.files = []
        if os.path.isdir(image_dir):
            for f in os.listdir(image_dir):
                if f.lower().endswith(IMG_EXTS):
                    self.files.append(f)

    def __len__(self):
        return len(self.files)

    def _load_clean(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if self.size:
            img = cv2.resize(img, (self.size[1], self.size[0]))
        return img

    def _degrade(self, img):
        """
        Orijinal goruntuyu taklit ederek bozulmus versiyon uretir:
        - Düsük isik (gamma < 1)
        - Gaussian noise
        - Blur (Gaussian)
        - Hafif downscale/upsample
        Tum bozulmalar rastgele secilir; her iterasyonda farkli kombinasyon olabilir.
        """
        out = img.copy()

        # Düsük isik: gamma < 1.0
        if random.random() < 0.7:
            gamma = random.uniform(0.4, 0.9)
            out = np.power(out, gamma)

        # Gaussian noise
        if random.random() < 0.7:
            sigma = random.uniform(0.0, 0.06)
            noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
            out = np.clip(out + noise, 0.0, 1.0)

        # Blur
        if random.random() < 0.5:
            k = random.choice([3, 5, 7])
            out = cv2.GaussianBlur(out, (k, k), 0)

        # Downscale / upscale (pixelation)
        if random.random() < 0.5:
            h, w = out.shape[:2]
            scale = random.uniform(0.5, 0.9)
            nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
            out_small = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
            out = cv2.resize(out_small, (w, h), interpolation=cv2.INTER_NEAREST)

        return out

    def __getitem__(self, idx):
        name = self.files[idx]
        path = os.path.join(self.image_dir, name)
        clean = self._load_clean(path)
        if clean is None:
            return self.__getitem__((idx + 1) % len(self.files))
        degraded = self._degrade(clean)

        clean_t = torch.from_numpy(clean).permute(2, 0, 1)
        degraded_t = torch.from_numpy(degraded).permute(2, 0, 1)
        return degraded_t, clean_t

