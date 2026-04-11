# Furkan-Develop Branch

**Owner:** Furkan Cabbar  
**Layer:** Global Enhancement — Exposure, Color, Contrast  
**Standalone repo:** https://github.com/furkancabbar/Global-Enhancement  
**Status:** In Development

---

## Responsibility

- Exposure balancing (histogram stretching, gamma correction)
- White balance correction (color cast removal)
- Contrast enhancement (adaptive tone mapping)
- Tint adjustment
- Night scene special handling (conservative — preserve colored lights)
- Skin-aware color correction (conservative around faces)

## Input From Pixel Layer (Batuhan)

```python
restored_image: np.ndarray   # GFPGAN-processed BGR uint8
profile: ProfileResult        # brightness, skin_ratio, is_night_scene, etc.
```

## Key Files

- `src/pipeline.py`          → `GlobalEnhancementPipeline`
- `src/exposure.py`          → `ExposureBalancer`
- `src/color_correction.py`  → `ColorCorrector`
- `src/contrast.py`          → `ContrastEnhancer`
- `src/profiler.py`          → `ImageProfiler`
- `src/metrics.py`           → `MetricsCalculator`
- `src/gui/main_window.py`   → PyQt6 GUI
