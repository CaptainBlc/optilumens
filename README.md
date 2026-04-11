# OptiLumen — Total-Develop: Full System Integration

> **Branch:** `Total-Develop` | **Team:** Furkan Cabbar · Batuhan Taşdemir · Ahmet Emir Ceylan  
> **Purpose:** Integration of all three enhancement layers

---

## Integration Architecture

```
Input Image (any device: phone, CCTV, drone, camera)
    │
    ├─── ImageProfiler.profile()          ← Scene analysis
    │
    ▼
┌────────────────────────────────────────────────┐
│            Semantic Layer (Furkan)             │
│   Face detection · Scene type · ROI mask       │
└───────────────────┬────────────────────────────┘
                    │  mask_map, face_bboxes, scene_flags
                    ▼
┌────────────────────────────────────────────────┐
│        Pixel Layer — GFPGAN v1.3 (Batuhan)    │
│   Face restoration · AI deblur · Denoise       │
│   Fidelity control · Explainability log        │
└───────────────────┬────────────────────────────┘
                    │  restored_image
                    ▼
┌────────────────────────────────────────────────┐
│          Global Layer (Emir)                   │
│   Exposure balance · White balance             │
│   Contrast · Gamma · Tint                      │
└───────────────────┬────────────────────────────┘
                    │
                    ▼
            Final Enhanced Image
            + Quality Metrics (PSNR, SSIM, ...)
            + Difference Heatmap
            + Processing Log
```

---

## Branch Status

| Layer | Branch | Status |
|-------|--------|--------|
| Semantic / Edge | `Furkan-Develop` | In development |
| Pixel (GFPGAN) | `Batuhan-Develop` | ✅ Active |
| Global Enhancement | `Emir-Develop` | In development |
| **Integration** | **`Total-Develop`** | **Pending all layers** |

---

## Setup

```bash
git clone https://github.com/CaptainBlc/CMPE491-AI-Camera.git
cd CMPE491-AI-Camera
git checkout Total-Develop
python setup.py
python src/gui_main.py
```

---

## Integration Example

```python
from profiler import ImageProfiler
from pipeline import ImageEnhancementPipeline
import cv2

image = cv2.imread("input.jpg")
profiler = ImageProfiler()
profile = profiler.profile(image)

# Step 1: Pixel restoration (GFPGAN)
pipe = ImageEnhancementPipeline(fidelity_weight=0.5)
pixel_result = pipe.restoreImage(image)

# Step 2: Global enhancement (Emir's module — plug in here)
# final = GlobalEnhancer().enhance(pixel_result.restored, profile)

final = pixel_result.restored  # until Global layer is integrated
cv2.imwrite("output.jpg", final)

print("\n".join(pixel_result.log))
```

---

*See individual layer branches: [`Batuhan-Develop`](../../tree/Batuhan-Develop) · [`Furkan-Develop`](../../tree/Furkan-Develop) · [`Emir-Develop`](../../tree/Emir-Develop) · [`legacy`](../../tree/legacy)*
