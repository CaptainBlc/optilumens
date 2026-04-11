# OptiLumen — Furkan-Develop: Global Enhancement Layer

> **Branch:** `Furkan-Develop` | **Owner:** Furkan Cabbar  
> **Layer:** Global Enhancement — Exposure, Color Balance, Contrast  
> **Furkan's repo:** [furkancabbar/Global-Enhancement](https://github.com/furkancabbar/Global-Enhancement)

---

## Layer Role

The Global Enhancement Layer applies **scene-wide** color and tonal corrections after GFPGAN face restoration. It handles everything that affects the whole image uniformly:

```
Pixel Layer output (GFPGAN restored faces)
    │
    ▼
┌──────────────────────────────────────────┐
│     Global Enhancement Layer (Furkan)   │
│  • Exposure / brightness balancing       │
│  • Gamma correction                      │
│  • White balance (color cast removal)    │
│  • Contrast & dynamic range              │
│  • Tint adjustment                       │
│  • Night scene detection & handling      │
└──────────────────────┬───────────────────┘
                       │
                       ▼
                Final Enhanced Image
```

---

## Critical Boundary with Pixel Layer

| | Pixel Layer (Batuhan) | Global Layer (Furkan) |
|--|----------------------|-----------------------|
| **Scope** | Face region only | Whole image |
| **Tech** | GFPGAN v1.3 (AI) | Tone/color mapping |
| **Brightness** | ❌ Never touches | ✅ Controls |
| **Color balance** | ❌ Never touches | ✅ Controls |
| **Face sharpness** | ✅ Face detail | ❌ |
| **Noise removal** | ✅ Face only | ❌ |

---

## Integration Point

```python
# Global layer receives GFPGAN-restored image:
from pipeline import ImageEnhancementPipeline

pixel_result = ImageEnhancementPipeline().restoreImage(image)
restored = pixel_result.restored   # GFPGAN output → feed into Global layer

# Furkan's module:
# final = GlobalEnhancementPipeline().enhance(restored, profile=pixel_result.profile)
```

---

## Key Classes (from Furkan's System)

| Class | File | Description |
|-------|------|-------------|
| `GlobalEnhancementPipeline` | `pipeline.py` | Main orchestrator |
| `ExposureBalancer` | `exposure.py` | Brightness & gamma |
| `ColorCorrector` | `color_correction.py` | White balance, tint |
| `ContrastEnhancer` | `contrast.py` | Dynamic range |
| `ImageProfiler` | `profiler.py` | Scene analysis |
| `MetricsCalculator` | `metrics.py` | PSNR, SSIM, heatmap |

---

## Setup

```bash
git clone https://github.com/CaptainBlc/CMPE491-AI-Camera.git
cd CMPE491-AI-Camera
git checkout Furkan-Develop
python setup.py
python src/gui_main.py
```

---

*Full project: [`main`](https://github.com/CaptainBlc/CMPE491-AI-Camera) · Furkan's standalone repo: [Global-Enhancement](https://github.com/furkancabbar/Global-Enhancement)*
