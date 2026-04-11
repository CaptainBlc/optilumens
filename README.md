# OptiLumen — Emir-Develop: Semantic / Edge Layer

> **Branch:** `Emir-Develop` | **Owner:** Ahmet Emir Ceylan  
> **Layer:** Semantic Understanding — Scene Analysis, Segmentation, Edge Detection

---

## Layer Role

The Semantic / Edge Layer is the **first processing stage**. It understands the scene before any enhancement begins, producing structured information that guides all downstream layers.

```
Raw Input Image
    │
    ▼
┌──────────────────────────────────────────┐
│     Semantic / Edge Layer (Emir)         │
│  • Face detection & localization         │
│  • Scene classification                  │
│    (portrait / landscape / night / ...)  │
│  • Segmentation mask (face, bg, sky...)  │
│  • Edge map generation                   │
│  • ROI importance mask (mask_map)        │
└──────────────────────┬───────────────────┘
                       │  mask_map, scene_flags, face_bboxes
                       ▼
┌──────────────────────────────────────────┐
│    Pixel Layer (Batuhan) — GFPGAN v1.3  │
└──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│    Global Enhancement (Furkan)           │
└──────────────────────────────────────────┘
```

---

## Output Contract

This layer must produce the following for downstream layers:

```python
@dataclass
class SemanticResult:
    mask_map: np.ndarray        # float32 HxW [0..1] — face/ROI importance map
    face_bboxes: List[tuple]    # [(x1,y1,x2,y2)] detected face regions
    scene_type: str             # "portrait" | "landscape" | "indoor" | "night"
    has_face: bool              # face detected?
    is_night_scene: bool        # night mode?
    edge_map: np.ndarray        # uint8 HxW — Canny edge map
    skin_ratio: float           # fraction of skin-tone pixels
```

---

## Integration with Pixel Layer

```python
# Emir's module produces semantic_result:
from pipeline import ImageEnhancementPipeline

pipe = ImageEnhancementPipeline()
result = pipe.restoreImage(
    image,
    mask_map=semantic_result.mask_map,   # Emir's ROI map guides GFPGAN
)
```

---

## Setup

```bash
git clone https://github.com/CaptainBlc/CMPE491-AI-Camera.git
cd CMPE491-AI-Camera
git checkout Emir-Develop
python setup.py
```

---

*Full project: [`main`](https://github.com/CaptainBlc/CMPE491-AI-Camera) · Furkan's global layer: [furkancabbar/Global-Enhancement](https://github.com/furkancabbar/Global-Enhancement)*
