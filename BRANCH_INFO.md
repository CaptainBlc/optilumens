# Furkan-Develop Branch

**Owner:** Furkan Cabbar  
**Layer:** Semantic / Edge Layer — Scene Understanding & Segmentation  
**Status:** In Development

---

## Layer Responsibility

The Semantic Layer is responsible for understanding **what** is in the image and **where**:

- Face detection and localization
- Scene classification (portrait, landscape, indoor, night, etc.)
- Segmentation mask generation (foreground / background / face region)
- Edge map generation
- Semantic label output for downstream layers

## Expected Output to Pixel Layer

```python
# This is what Batuhan's Pixel Layer (GFPGAN) expects:
mask_map: np.ndarray    # float32 [0..1], HxW — face/ROI importance map
scene_flags: dict       # e.g. {"is_night": False, "has_face": True, ...}
face_bboxes: list       # [(x1,y1,x2,y2), ...] detected face regions
```

## Integration Point

```python
# How Furkan's module connects to Batuhan's pipeline:
from pipeline import ImageEnhancementPipeline

semantic_result = furkan_semantic_layer.analyze(image)
pipe = ImageEnhancementPipeline()
result = pipe.restoreImage(
    image,
    face_bboxes=semantic_result.face_bboxes,
    mask_map=semantic_result.mask_map,
)
```

## Files in This Branch

- `src/semantic/` — Semantic layer module (to be implemented)
- `src/edge_detector.py` — Edge detection module
- `src/segmenter.py` — Image segmentation
