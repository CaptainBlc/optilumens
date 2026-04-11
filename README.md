# OptiLumen — Furkan-Develop: Semantic / Edge Layer

> **Branch:** `Furkan-Develop` | **Owner:** Furkan Cabbar  
> **Layer:** Semantic Understanding — Scene Analysis & Segmentation

---

## Layer Role

The Semantic Layer is the **first stage** in the pipeline. It processes the raw input image to produce structured information that guides all downstream layers (Pixel, Global).

```
Input Image
    │
    ▼
┌─────────────────────────────┐
│     Semantic Layer (Furkan) │
│  • Face detection           │
│  • Scene classification     │
│  • Segmentation mask        │
│  • Edge map                 │
└─────────────┬───────────────┘
              │  mask_map, scene_flags, face_bboxes
              ▼
┌─────────────────────────────┐
│    Pixel Layer (Batuhan)    │ ← GFPGAN v1.3 face restoration
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│    Global Layer (Emir)      │ ← exposure, color, contrast
└─────────────────────────────┘
```

---

## Setup

```bash
git clone https://github.com/CaptainBlc/CMPE491-AI-Camera.git
cd CMPE491-AI-Camera
git checkout Furkan-Develop
python setup.py
```

---

## Output Contract

This layer must produce the following for downstream layers:

```python
@dataclass
class SemanticResult:
    mask_map: np.ndarray        # float32 HxW [0..1] face/ROI importance
    face_bboxes: List[tuple]    # [(x1,y1,x2,y2)] face locations
    scene_type: str             # "portrait" | "landscape" | "indoor" | "night"
    has_face: bool
    is_night_scene: bool
    edge_map: np.ndarray        # uint8 HxW Canny edges
```

---

## Integration with Pixel Layer

```python
from pipeline import ImageEnhancementPipeline

# Furkan's module produces semantic_result
pipe = ImageEnhancementPipeline()
result = pipe.restoreImage(image, mask_map=semantic_result.mask_map)
```

---

*For the full project, see the [`main`](https://github.com/CaptainBlc/CMPE491-AI-Camera) branch.*
