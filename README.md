# OptiLumen — Total-Develop: Full System Integration

> **Branch:** `Total-Develop` | **Team:** Furkan Cabbar · Batuhan Taşdemir · Ahmet Emir Ceylan  
> **Purpose:** Integration of all three enhancement layers into one unified pipeline

---

## Full System Architecture

```
Input Image (any device: phone, CCTV, drone, camera)
    │
    ├─── ImageProfiler.profile()          [shared utility]
    │
    ▼
┌────────────────────────────────────────────────────┐
│        Semantic / Edge Layer  (Emir)               │
│   Face detection · Scene classification            │
│   Segmentation mask (mask_map) · Edge map          │
└─────────────────────┬──────────────────────────────┘
                      │  mask_map, face_bboxes, scene_flags
                      ▼
┌────────────────────────────────────────────────────┐
│      Pixel Layer — GFPGAN v1.3  (Batuhan)         │
│   AI face restoration · Deblur · Denoise           │
│   Fidelity control (0–100%) · Explainability log   │
│   → branch: Batuhan-Develop                        │
└─────────────────────┬──────────────────────────────┘
                      │  restored_image
                      ▼
┌────────────────────────────────────────────────────┐
│      Global Enhancement  (Furkan)                  │
│   Exposure · Gamma · White balance                 │
│   Contrast · Tint · Night scene handling           │
│   → repo: furkancabbar/Global-Enhancement          │
└─────────────────────┬──────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────┐
    │      Final Output           │
    │  + PSNR / SSIM metrics      │
    │  + Difference heatmap       │
    │  + Full processing log      │
    └─────────────────────────────┘
```

---

## Branch & Repo Map

| Layer | Developer | Branch | Repo |
|-------|-----------|--------|------|
| Semantic / Edge | Emir Ceylan | [`Emir-Develop`](../../tree/Emir-Develop) | This repo |
| Pixel (GFPGAN) | Batuhan Taşdemir | [`Batuhan-Develop`](../../tree/Batuhan-Develop) | This repo |
| Global Enhancement | Furkan Cabbar | [`Furkan-Develop`](../../tree/Furkan-Develop) | [Global-Enhancement](https://github.com/furkancabbar/Global-Enhancement) |

---

## Integration Code

```python
import cv2
from profiler import ImageProfiler
from pipeline import ImageEnhancementPipeline

image = cv2.imread("input.jpg")

# Step 0: Scene analysis
profile = ImageProfiler().profile(image)

# Step 1: Semantic layer (Emir) — provides mask_map to guide GFPGAN
# semantic = EmirsSemanticLayer().analyze(image)
# mask_map = semantic.mask_map

# Step 2: Pixel layer (Batuhan) — GFPGAN v1.3 face restoration
pipe = ImageEnhancementPipeline(fidelity_weight=0.5)
pixel_result = pipe.restoreImage(image)  # mask_map=semantic.mask_map when integrated

# Step 3: Global enhancement (Furkan) — tone & color
# from global_enhancement import GlobalEnhancementPipeline
# final = GlobalEnhancementPipeline().enhance(pixel_result.restored, profile=profile)

final = pixel_result.restored   # temporary until all layers integrated

cv2.imwrite("output.jpg", final)
print("\n".join(pixel_result.log))
```

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

## Status Dashboard

| Component | Status | Notes |
|-----------|--------|-------|
| Pixel Layer (GFPGAN v1.3) | ✅ Working | Download model: `python scripts/download_model.py` |
| Global Enhancement | 🔄 In progress | [Furkan's repo](https://github.com/furkancabbar/Global-Enhancement) |
| Semantic / Edge Layer | 🔄 In progress | Emir's module |
| Full integration | ⏳ Pending | Waiting for all layers |
| GUI | ✅ Working | `python src/gui_main.py` |

---

*Individual branches: [`Batuhan-Develop`](../../tree/Batuhan-Develop) · [`Furkan-Develop`](../../tree/Furkan-Develop) · [`Emir-Develop`](../../tree/Emir-Develop) · [`legacy`](../../tree/legacy)*
