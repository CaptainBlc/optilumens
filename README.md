# OptiLumen — Legacy: Classical Pixel Enhancement System

> **⚠️ This is the `legacy` branch** — the baseline system before GFPGAN integration.  
> For the current AI-based system, see the [`main`](https://github.com/CaptainBlc/CMPE491-AI-Camera) branch.

---

## What Is This?

This branch documents the **Phase 1 development** of the OptiLumen image enhancement system.  
It uses classical OpenCV-based image processing — no deep learning.

It serves as both a **technical baseline** and a **development history reference**, showing how the project evolved before the professor recommended GFPGAN v1.3.

---

## Development Timeline (This Branch)

```
v0.1 — Basic Pipeline
  main.py + profiler.py + pixel_enhance.py
  └── brightness boost, blur sharpening

v0.2 — Edge Enhancement
  + edge_enhance.py (Canny-based)

v0.3 — Mask-Based ROI Processing
  + mask_map support (ROI = face/center, BG = background)
  + pixel_pipeline.py (ImageProcessor class)

v0.4 — Professional Refactor
  + PixelBasedProcessor class (aligned with HLD Report)
  + reduceNoise() / sharpenImage() / edgePreservingFilter()
  + Input validation, type hints, docstrings

v0.5 — AI Integration (EnhanceNet)
  + EnhanceNet: lightweight U-Net CNN
  + Self-supervised training dataset
  + Decision Engine (adaptive parameter selection)
  + Quality Metrics (PSNR, SSIM, difference heatmap)

v0.6 — PyQt6 GUI + Swipe Compare
  + Interactive GUI with fidelity slider
  + Before/After swipe compare widget
  + Processing log (explainability)

  ↓ Professor recommends GFPGAN v1.3
  
→ See main / Batuhan-Develop for GFPGAN system
```

---

## Running the Legacy System

```bash
# Install (no PyTorch needed for classical mode)
pip install opencv-python numpy PyQt6

# GUI
cd src
python gui_main.py

# Batch processing
python mask_test_batch.py
```

---

## Key Files

| File | Version | Description |
|------|---------|-------------|
| `legacy/pixel_enhance.py` | v0.4+ | `PixelBasedProcessor` — adaptive sharpening, CLAHE, bilateral |
| `legacy/pixel_pipeline.py` | v0.3+ | `ImageProcessor` — pipeline orchestrator |
| `legacy/profiler.py` | v0.2+ | `ImageProfiler` — brightness, blur, noise, skin |
| `legacy/metrics.py` | v0.5+ | PSNR, SSIM, Entropy, difference heatmap |
| `legacy/models/enhance_net.py` | v0.5 | EnhanceNet — lightweight U-Net CNN |
| `legacy/train.py` | v0.5 | Self-supervised training |
| `legacy/gui/main_window.py` | v0.6 | PyQt6 GUI |
| `legacy/mask_test_batch.py` | v0.5 | Batch testing with ROI masks |

---

## Why We Migrated to GFPGAN

| Classical (this branch) | GFPGAN v1.3 (main) |
|------------------------|---------------------|
| OpenCV sharpening + CLAHE | StyleGAN2 generative prior |
| Rule-based decisions | Neural network inference |
| No face detection | RetinaFace + landmark alignment |
| PSNR 47–51 dB (very subtle) | Blind face restoration |
| Can't recover severe blur | Recovers severe degradation |
| ~5 MB code, no GPU needed | 340 MB model, GPU optional |

---

## Interface Contract (Pixel Layer)

This interface is preserved in the GFPGAN branch too:

```python
# Input
img: np.ndarray          # BGR uint8, any resolution
mask_map: np.ndarray     # float32 [0..1], HxW (optional)
profile: dict            # from ImageProfiler

# Output
result: np.ndarray       # BGR uint8, same resolution
log: list[str]           # step-by-step explanation
metrics: QualityMetrics  # PSNR, SSIM, etc.
```
