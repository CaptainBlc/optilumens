"""
Pixel iyilestirme agini eslenmis veri (girdi -> hedef) uzerinde egitir.
Kullanim: python train.py
Veri: data/train/input/ ve data/train/target/ (ayni dosya isimleri).
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import EnhanceNet
from dataset import PairedImageDataset, SelfSupervisedImageDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_INPUT_DIR = "../data/train/input"   # varsa paired icin
DEFAULT_TARGET_DIR = "../data/train/target" # varsa paired icin
DEFAULT_UNPAIRED_DIR = "../data/train/images"  # yoksa tek klasor icin
CHECKPOINT_DIR = "../checkpoints"
BATCH = 4
EPOCHS = 50
LR = 1e-3
CROP = 256  # egitimde rastgele 256x256 kirpma (opsiyonel)


def main():
    p = argparse.ArgumentParser(description="Train EnhanceNet (pixel enhancement)")
    p.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Low/input images (paired) or images dir (unpaired)")
    p.add_argument("--target_dir", default=DEFAULT_TARGET_DIR, help="High/target images (paired only)")
    p.add_argument("--checkpoints", default=CHECKPOINT_DIR, help="Save dir")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--size", type=int, default=CROP, help="Train crop size (0 = full)")
    args = p.parse_args()

    os.makedirs(args.checkpoints, exist_ok=True)
    size = (args.size, args.size) if args.size else None

    # 1) Once paired veri var mi kontrol et
    ds = PairedImageDataset(args.input_dir, args.target_dir, size=size)
    mode = "paired"

    # 2) Paired yoksa self-supervised moda gec: tek klasorden bozulmus->temiz ogrenme
    if len(ds) == 0:
        # Kullanici input_dir vermisse onu, yoksa varsayilan unpaired klasorunu kullan
        unpaired_dir = args.input_dir
        if not any(f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"))
                   for f in os.listdir(unpaired_dir) if os.path.isfile(os.path.join(unpaired_dir, f))):
            unpaired_dir = DEFAULT_UNPAIRED_DIR
        ds = SelfSupervisedImageDataset(unpaired_dir, size=size)
        mode = "self_supervised"

    if len(ds) == 0:
        print("No training images found.")
        print("Paired mode: put same-named files into", DEFAULT_INPUT_DIR, "and", DEFAULT_TARGET_DIR)
        print("Self-supervised mode: put arbitrary images into", DEFAULT_UNPAIRED_DIR)
        return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)

    net = EnhanceNet(3, 3).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    print("Training mode:", mode, "| images:", len(ds))

    for epoch in range(1, args.epochs + 1):
        net.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        print("Epoch", epoch, "| loss:", round(avg, 5))
        if epoch % 10 == 0 or epoch == args.epochs:
            path = os.path.join(args.checkpoints, "enhance_net_latest.pt")
            torch.save({"state_dict": net.state_dict(), "epoch": epoch}, path)
            print("  Saved", path)
    print("Training done. Checkpoints in", args.checkpoints)
