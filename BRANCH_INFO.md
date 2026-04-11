# Emir-Develop Branch

**Owner:** Ahmet Emir Ceylan  
**Layer:** Global Enhancement — Exposure, Color Balance, Contrast  
**Status:** In Development

---

## Layer Responsibility

- Exposure balancing (histogram stretching, gamma correction)
- White balance correction (color cast removal)
- Contrast enhancement (adaptive tone mapping)
- Tint adjustment
- Night scene handling

## Input From Pixel Layer

```python
# Emir receives:
restored_image: np.ndarray   # GFPGAN-processed BGR uint8 image
profile: ProfileResult        # scene analysis (brightness, skin, night, etc.)
```

## Files

- `src/global_enhancer.py` — GlobalEnhancer class
- `src/exposure.py` — ExposureBalancer
- `src/color_correction.py` — ColorCorrector
- `src/contrast.py` — ContrastEnhancer
