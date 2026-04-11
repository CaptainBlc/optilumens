# Emir-Develop Branch

**Owner:** Ahmet Emir Ceylan  
**Layer:** Semantic / Edge Layer — Scene Understanding & Segmentation  
**Status:** In Development

---

## Responsibility

- Face detection and bounding box localization
- Scene classification (portrait, landscape, indoor, night, etc.)
- Segmentation mask generation (foreground/background/face region)
- ROI importance map (`mask_map`) for the Pixel Layer
- Edge map generation (Canny / Sobel)
- Skin-tone detection

## Output to Pixel Layer

```python
mask_map: np.ndarray      # float32 [0..1] HxW — high=face/ROI, low=background
face_bboxes: list         # [(x1,y1,x2,y2), ...]
scene_flags: dict         # {"is_night": bool, "has_face": bool, ...}
edge_map: np.ndarray      # uint8 HxW
```

## Planned Files

- `src/semantic/segmenter.py`     → Segmentation module
- `src/semantic/edge_detector.py` → Edge detection
- `src/semantic/scene_classifier.py` → Scene type classification
- `src/semantic/face_detector.py` → Lightweight face detection
