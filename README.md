# OptiLumen — Emir-Develop: Global Enhancement Layer

> **Branch:** `Emir-Develop` | **Owner:** Ahmet Emir Ceylan  
> **Layer:** Global Enhancement — Exposure, Color, Contrast

---

## Layer Role

The Global Layer applies **scene-wide** adjustments after face restoration. It handles everything that affects the whole image uniformly:

```
Pixel Layer output (GFPGAN restored faces)
    │
    ▼
┌──────────────────────────────────┐
│    Global Enhancement (Emir)     │
│  • Exposure / brightness balance │
│  • Gamma correction              │
│  • White balance / color cast    │
│  • Contrast (dynamic range)      │
│  • Tint adjustment               │
└──────────────────┬───────────────┘
                   │
                   ▼
            Final Output Image
```

---

## Critical Boundary with Pixel Layer

| | Pixel Layer (Batuhan) | Global Layer (Emir) |
|--|----------------------|---------------------|
| **Scope** | Face region only | Whole image |
| **Operation** | AI face restoration | Tone / color mapping |
| **Brightness** | ❌ Never touches | ✅ Controls |
| **Color** | ❌ Never touches | ✅ Controls |
| **Sharpening** | ✅ Face detail | ❌ |
| **Noise** | ✅ Face only | ❌ |

---

## Integration Point

```python
# Global layer receives GFPGAN-restored image:
from pipeline import ImageEnhancementPipeline

pixel_result = ImageEnhancementPipeline().restoreImage(image)
restored = pixel_result.restored   # GFPGAN output

# Emir's module processes this:
final = emir_global_layer.enhance(restored, profile=pixel_result.profile)
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

*For the full project, see the [`main`](https://github.com/CaptainBlc/CMPE491-AI-Camera) branch.*
