# Batuhan-Develop Branch

**Owner:** Batuhan Taşdemir  
**Layer:** Pixel Layer — GFPGAN v1.3 Face Restoration  
**Status:** Active Development

---

## This Branch Contains

The complete **Pixel Layer** implementation using GFPGAN v1.3:

- `src/models/gfpgan_arch.py` — GFPGANv1Clean (StyleGAN2-based)
- `src/models/stylegan2_clean.py` — StyleGAN2 backbone, no custom CUDA ops
- `src/face_restorer.py` — `FaceRestorer` class (main module)
- `src/pipeline.py` — `ImageEnhancementPipeline` orchestrator
- `src/profiler.py` — `ImageProfiler` for scene analysis
- `src/metrics.py` — quality evaluation
- `src/gui/main_window.py` — interactive PyQt6 GUI

## Pixel Layer Responsibility

Per the HLD Report, this layer handles **only**:
- Face detection (RetinaFace via facexlib)
- AI face restoration (GFPGAN v1.3)
- Fidelity-controlled blending (user-adjustable 0–100%)
- Quality metrics & explainability log

**NOT** responsible for: global brightness, color balance, contrast (→ Emir's Global Layer)

## Interface with Other Layers

```python
from pipeline import ImageEnhancementPipeline

pipe = ImageEnhancementPipeline(fidelity_weight=0.5)
result = pipe.restoreImage(image_bgr)
# result.restored  → enhanced image (numpy BGR)
# result.metrics   → PSNR, SSIM, etc.
# result.log       → step-by-step explanation
```

## Setup

```bash
python setup.py      # installs deps + downloads GFPGANv1.3.pth
python src/gui_main.py
```
