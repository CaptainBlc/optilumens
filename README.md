# OptiLumen — Hybrid AI Image Enhancement System
### CMPE 491 Senior Design Project | Çankaya University

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![PyQt6](https://img.shields.io/badge/PyQt6-6.5%2B-green?logo=qt)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**Team:** Furkan Cabbar · Batuhan Taşdemir · Ahmet Emir Ceylan

[Website](https://optilumen.netlify.app) · [Docs](./docs/) · [Legacy System](https://github.com/CaptainBlc/CMPE491-AI-Camera/tree/legacy)

</div>

---

## What is OptiLumen?

OptiLumen is a multi-layer hybrid image enhancement system designed for device-independent face restoration. It combines AI-based face restoration (GFPGAN v1.3) with classical image processing to produce high-quality, explainable enhancements on images from any source — phones, cameras, CCTV, drones.

```
Input Image
    │
    ├─► Semantic Layer   (scene understanding, face detection)
    ├─► Pixel Layer      ← Batuhan's module: GFPGAN v1.3 face restoration
    ├─► Global Layer     ← Emir's module: exposure, color, contrast
    └─► Fusion / Output
```

---

## Branch Structure

| Branch | Owner | Description |
|--------|-------|-------------|
| `main` | Team | Stable integration — what you're reading now |
| `Batuhan-Develop` | Batuhan Taşdemir | **Pixel Layer** — GFPGAN v1.3 face restoration |
| `Furkan-Develop` | Furkan Cabbar | **Semantic / Edge Layer** — scene analysis |
| `Emir-Develop` | Ahmet Emir Ceylan | **Global Layer** — exposure, color, contrast |
| `Total-Develop` | Team | Integration of all three layers |
| `legacy` | — | Classical pixel enhancement (pre-GFPGAN baseline) |

---

## Quick Start (Clone & Run)

### 1 — Clone

```bash
git clone https://github.com/CaptainBlc/CMPE491-AI-Camera.git
cd CMPE491-AI-Camera
```

### 2 — Install & Download Model (one command)

```bash
python setup.py
```

This installs all dependencies **and** downloads the GFPGANv1.3 model weights (~340 MB) automatically.

### 3 — Launch GUI

```bash
python src/gui_main.py
```

### Manual steps (if setup.py fails)

```bash
# Install dependencies
pip install -r requirements.txt

# Download model weights
python scripts/download_model.py

# Launch
python src/gui_main.py
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11 |
| RAM | 4 GB | 8 GB |
| GPU | — (CPU works) | NVIDIA CUDA 11+ |
| Disk | 500 MB free | 1 GB |
| OS | Windows 10+ / Linux / macOS | — |

> **Note:** On Python 3.13, `torch` may require the CPU-only build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Project Structure

```
CMPE491-AI-Camera/
├── src/
│   ├── models/
│   │   ├── gfpgan_arch.py      # GFPGANv1Clean architecture
│   │   └── stylegan2_clean.py  # StyleGAN2 backbone (pure PyTorch)
│   ├── face_restorer.py        # FaceRestorer — GFPGAN v1.3 wrapper
│   ├── pipeline.py             # ImageEnhancementPipeline (orchestrator)
│   ├── profiler.py             # ImageProfiler — scene analysis
│   ├── metrics.py              # PSNR, SSIM, Entropy, Colorfulness
│   ├── main.py                 # Batch processing CLI
│   ├── gui_main.py             # GUI entry point
│   └── gui/
│       └── main_window.py      # PyQt6 interactive GUI
├── scripts/
│   └── download_model.py       # GFPGANv1.3.pth downloader
├── checkpoints/                # Model weights (not in git, auto-downloaded)
├── inputs/                     # Input images for batch mode
├── outputs/                    # Enhanced outputs
├── docs/                       # Design reports, specifications
├── legacy/                     # Classical baseline system (see legacy branch)
├── setup.py                    # One-command setup
└── requirements.txt
```

---

## GUI Overview

The desktop GUI (PyQt6) provides:

- **Drag & drop** image loading
- **Fidelity slider** — control how much AI vs. original (0% = full GFPGAN, 100% = no change)
- **Interactive swipe compare** — drag divider to compare before/after
- **Quality metrics** — PSNR, SSIM, Entropy, Colorfulness (before & after)
- **Difference heatmap** — visual explanation of where changes were applied
- **Processing log** — step-by-step explainability output

---

## How GFPGAN Works (Pixel Layer)

```
Input Face (512×512)
    │
    ├─ U-Net Encoder     → extracts degradation features
    ├─ StyleGAN2 Prior   → generates clean face structure
    ├─ SFT Conditions    → spatial feature transform (detail guidance)
    └─ Fidelity Blend    → (1-α)×GFPGAN + α×original
         │
    Restored Face
```

The model was pre-trained on FFHQ (70,000 high-quality faces) and fine-tuned for blind face restoration — handling blur, noise, low-light, and compression artifacts simultaneously.

---

## Documentation

- [High Level Design Report](./docs/High_Level_Design_Report.md)
- [Project Specifications](./docs/Project_Specifications_Report.md)
- [Analysis Report](./docs/Analysis_Report.md)
- [Pixel Layer Interface Specification](./docs/TASARIM_OZETI.md)

---

## License

This project is developed for academic purposes as part of CMPE 491 Senior Design Project at Çankaya University.  
GFPGAN architecture reproduced under [MIT License](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE).
