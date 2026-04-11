# Legacy — Classical Pixel Enhancement System

> **This is the baseline system** developed before the GFPGAN v1.3 integration.  
> It represents Phase 1 of the OptiLumen project (classical image processing approach).

---

## What This Is

This branch archives the original classical pixel enhancement pipeline built during the first phase of CMPE 491. It uses no deep learning — only OpenCV-based image processing.

Think of this as the **"before"** in the project's development story:

```
Legacy (this branch)              →   Main / Batuhan-Develop
─────────────────────────────────────────────────────────────
Classical OpenCV pipeline         →   GFPGAN v1.3 (AI)
Rule-based decisions              →   Neural network decisions
No face detection                 →   RetinaFace detection
PSNR ~47-51 dB (subtle)          →   Full blind face restoration
pixel_enhance() function          →   FaceRestorer class
mask_map region processing        →   SFT spatial conditioning
```

---

## Key Files

| File | Description |
|------|-------------|
| `src/pixel_enhance.py` | Core: adaptive sharpening, CLAHE, bilateral filter |
| `src/pixel_pipeline.py` | `ImageProcessor` orchestrator class |
| `src/profiler.py` | `ImageProfiler` — brightness, blur, noise, skin detection |
| `src/metrics.py` | PSNR, SSIM, Entropy, Colorfulness, Difference heatmap |
| `src/mask_test_batch.py` | Batch testing with ROI mask |
| `src/gui/main_window.py` | PyQt6 GUI (first generation) |
| `src/dataset.py` | Self-supervised training dataset |
| `src/train.py` | EnhanceNet training script |

---

## Running the Legacy System

```bash
# Install dependencies
pip install opencv-python numpy PyQt6

# Batch processing
cd src
python mask_test_batch.py

# GUI
python gui_main.py
```

---

## Why We Moved to GFPGAN

| Limitation | Impact |
|-----------|--------|
| No face-specific detection | Treats face pixels same as background |
| Rule-based parameter selection | Cannot adapt to unknown degradation types |
| Classical sharpening only | Cannot restore severely blurred/noisy faces |
| PSNR 47-51 dB → very subtle | Changes barely perceptible |
| No generative prior | Cannot "hallucinate" missing detail |

GFPGAN solves all of these by using a StyleGAN2 generative prior trained on 70,000 high-quality faces (FFHQ), enabling **blind face restoration** — recovery from blur, noise, compression artifacts, and low-light simultaneously.

---

## Development Timeline

```
Phase 1 (Legacy)
  ├── pixel_enhance.py  — basic sharpening + brightness
  ├── profiler.py       — image analysis
  ├── edge_enhance.py   — Canny-based edge enhancement
  └── orchestrator.py   — pipeline coordination
       │
       ▼ (Major refactor)
  ├── PixelBasedProcessor class  (aligned with HLD report)
  ├── EnhanceNet (lightweight CNN)
  ├── Decision Engine
  └── PyQt6 GUI (interactive)
       │
       ▼ (Professor recommendation: GFPGAN v1.3)
Phase 2 (main / Batuhan-Develop)
  └── Full GFPGAN v1.3 integration
```
