# OptiLumen — Hybrid AI Image Enhancement System
### CMPE 491 Senior Design Project | TED University

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
    ├─► Semantic / Edge Layer  ← Emir's module: scene understanding, segmentation
    ├─► Pixel Layer            ← Batuhan's module: GFPGAN v1.3 face restoration
    ├─► Global Enhancement     ← Furkan's module: exposure, color, contrast
    └─► Fusion / Output
```

---

## Branch Structure

| Branch | Owner | Description |
|--------|-------|-------------|
| `main` | Team | Stable integration — what you're reading now |
| `Batuhan-Develop` | Batuhan Taşdemir | **Pixel Layer** — GFPGAN v1.3 face restoration |
| `Furkan-Develop` | Furkan Cabbar | **Global Enhancement Layer** — exposure, color, contrast · [Furkan's repo](https://github.com/furkancabbar/Global-Enhancement) |
| `Emir-Develop` | Ahmet Emir Ceylan | **Semantic / Edge Layer** — scene analysis, segmentation |
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
│   ├── camera_capture.py       # Live webcam I/O (Scenario 4)
│   ├── batch_processor.py      # Background batch worker (Scenarios 1 & 2)
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
- **Live camera mode** — real-time preview & one-click capture (Scenario 4)
- **Batch folder processing** — non-blocking queue with per-file status + cancel (Scenarios 1 & 2)
- **Fidelity slider** — control how much AI vs. original (0% = full GFPGAN, 100% = no change)
- **Interactive swipe compare** — drag divider to compare before/after
- **Quality metrics** — PSNR, SSIM, Entropy, Colorfulness (before & after)
- **Difference heatmap** — visual explanation of where changes were applied
- **Processing log** — step-by-step explainability output

### Live Camera Mode (Analysis Report §3.5.1 — Scenario 4)

Click **● Live** in the toolbar to stream your webcam feed into the GUI. Press **Capture** to take a stabilised snapshot, which is fed straight into the GFPGAN pipeline as if it were a loaded file. Works with any DirectShow / V4L2 compatible camera (laptop webcam, USB webcam, phone via DroidCam). No camera? The toolbar simply stays inactive — no failure mode leaks into the rest of the UI.

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

## Continuing Development on Another PC

This project is designed so you can clone it on any machine and have an AI assistant (Cursor / Claude Code) instantly pick up where the last session left off.

- **[`PROGRESS.md`](./PROGRESS.md)** — running development log. Humans and AI both read this to see current sprint, decisions, and next actions.
- **[`.cursor/rules/optilumen.mdc`](./.cursor/rules/optilumen.mdc)** — Cursor rule file that is auto-loaded by the AI on every session. Contains team rules, architecture contracts, commit workflow.

After any non-trivial change, update the **Changelog** section of `PROGRESS.md` and push. That is the handoff.

```bash
git clone https://github.com/CaptainBlc/Global-Enhancement.git
cd Global-Enhancement
py -3 -m venv venv && venv\Scripts\activate
python setup.py           # installs deps + downloads model
```

---

## Roadmap

Derived from the [Analysis Report §3.5.1](./docs/Analysis_Report.md) use-case scenarios:

| Sprint | Feature | Status | Scenario |
|--------|---------|--------|----------|
| 1 | Real-time camera capture | **Done** | §3.5.1 #4 |
| 2 | Background rendering (batch queue, non-blocking GUI) | **Done** | §3.5.1 #1,#2 |
| 3 | AI chat prompt for photo editing ("make it warmer", "soften skin") | Planned | §3.5.1 #3 |
| 4 | Live video filter stream (Snapchat/Instagram-style) | Planned | §3.5.1 #4 ext. |

---

## License

This project is developed for academic purposes as part of CMPE 491 Senior Design Project at TED University.  
GFPGAN architecture reproduced under [MIT License](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE).
