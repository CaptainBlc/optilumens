# OptiLumen — Hybrid AI Image Enhancement System
### CMPE 491 Senior Design Project | TED University

> **Integration:** `main` combines the **Pixel Layer** (GFPGAN), **semantic face parsing + region enhancement** (merged from `Furkan-Develop`), and leaves a hook for **Global Enhancement** (exposure, color, contrast).  
> **Furkan's standalone work:** [furkancabbar/Global-Enhancement](https://github.com/furkancabbar/Global-Enhancement)

---

## Layer Role

End-to-end flow:

```
Input
    │
    ├─► Semantic / Edge Layer  ← scene understanding, segmentation (team contract)
    ├─► Pixel Layer            ← GFPGAN v1.3 face restoration
    ├─► Face parse + regions   ← `FaceParser` + `RegionEnhancer` (in-repo, wired in `pipeline.py`)
    ├─► Global Enhancement     ← exposure, color, contrast (placeholder in pipeline log)
    └─► Fusion / Output
```

**Global Enhancement (whole-image)** handles scene-wide tone and color. **Region enhancement** in this repo refines skin, eyes, lips, etc. after GFPGAN using parsed masks.

---

## Branch layout

| Branch | Owner | Description |
|--------|-------|-------------|
| `main` | Team | Stable integration — what you're reading now |
| `Batuhan-Develop` | Batuhan Taşdemir | **Pixel Layer** — GFPGAN v1.3 face restoration |
| `Furkan-Develop` | Furkan Cabbar | **Global / integration experiments** — region pipeline, docs |
| `Emir-Develop` | Ahmet Emir Ceylan | **Semantic / Edge Layer** — scene analysis, segmentation |
| `Total-Develop` | Team | Integration of all three layers |
| `legacy` | — | Classical pixel enhancement (pre-GFPGAN baseline) |

| | Pixel Layer (Batuhan) | Region / global styling |
|--|----------------------|-------------------------|
| **Scope** | Face restoration (GFPGAN) | Per-face regions + (future) whole-image global |
| **Brightness (global)** | Not the primary role | Planned Global layer |
| **Face detail** | ✅ | ✅ (semantic masks) |

---

## Integration point

```python
from pipeline import ImageEnhancementPipeline

pipeline = ImageEnhancementPipeline()
result = pipeline.restoreImage(image)
# result.restored — output after GFPGAN + optional region enhancement
# result.semantic, result.region_result — parsing / region apply details
```

### Key classes (current `src`)

| Class | Module | Description |
|-------|--------|-------------|
| `ImageEnhancementPipeline` | `pipeline.py` | Orchestrator |
| `FaceParser` | `semantic_parser.py` | Face parsing → `SemanticResult` |
| `RegionEnhancer` | `region_enhancer.py` | Per-region enhancement from masks |
| `FaceRestorer` | `face_restorer.py` | GFPGAN wrapper |
| `ImageProfiler` | `profiler.py` | Scene analysis |
| `MetricsCalculator` | `metrics.py` | PSNR, SSIM, heatmap |

---

## Setup

```bash
git clone https://github.com/CaptainBlc/Global-Enhancement.git
cd Global-Enhancement
py -3 -m venv venv
venv\Scripts\activate
python setup.py
python src/gui_main.py
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| RAM | 4 GB | 8 GB |
| GPU | — (CPU works) | NVIDIA CUDA 11+ |
| Disk | 500 MB free | 1 GB |
| OS | Windows 10+ / Linux / macOS | — |

> **Note:** On Python 3.13, `torch` may require the CPU-only build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Project Structure

```
Global-Enhancement/
├── src/
│   ├── models/
│   │   ├── gfpgan_arch.py
│   │   └── stylegan2_clean.py
│   ├── face_restorer.py
│   ├── pipeline.py
│   ├── profiler.py
│   ├── metrics.py
│   ├── semantic_parser.py      # Face parsing (facexlib / ParseNet)
│   ├── region_enhancer.py      # Per-region enhancement
│   ├── live_capture.py         # Live capture helper (Furkan branch)
│   ├── camera_capture.py
│   ├── batch_processor.py
│   ├── chat_commands.py
│   ├── live_filters.py
│   ├── main.py
│   ├── gui_main.py
│   └── gui/
│       └── main_window.py
├── scripts/
│   └── download_model.py
├── checkpoints/
├── outputs/
├── docs/
├── legacy/
├── setup.py
└── requirements.txt
```

---

## GUI Overview

The desktop GUI (PyQt6) provides:

- **Drag & drop** image loading
- **Live camera mode** — real-time preview & capture
- **Live video filters** — OFF / BEAUTY / AI during live preview
- **Batch folder processing** — background queue with cancel
- **Chat bar** — rule-based commands (TR + EN)
- **Fidelity slider** — AI vs. original blend
- **Swipe compare**, **metrics**, **difference heatmap**, **processing log**

---

## How GFPGAN Works (Pixel Layer)

```
Input Face (512×512)
    │
    ├─ U-Net Encoder     → extracts degradation features
    ├─ StyleGAN2 Prior   → generates clean face structure
    ├─ SFT Conditions    → spatial feature transform
    └─ Fidelity Blend    → (1-α)×GFPGAN + α×original
         │
    Restored Face
```

---

## Documentation

- [High Level Design Report](./docs/High_Level_Design_Report.md)
- [Project Specifications](./docs/Project_Specifications_Report.md)
- [Analysis Report](./docs/Analysis_Report.md)
- [Pixel Layer Interface Specification](./docs/TASARIM_OZETI.md)

---

## Continuing Development on Another PC

- **[`PROGRESS.md`](./PROGRESS.md)** — development log
- **[`.cursor/rules/optilumen.mdc`](./.cursor/rules/optilumen.mdc)** — Cursor rules

```bash
git clone https://github.com/CaptainBlc/Global-Enhancement.git
cd Global-Enhancement
py -3 -m venv venv
venv\Scripts\activate
python setup.py
```

---

## Roadmap

| Sprint | Feature | Status | Scenario |
|--------|---------|--------|----------|
| 1 | Real-time camera capture | **Done** | §3.5.1 #4 |
| 2 | Background batch rendering | **Done** | §3.5.1 #1,#2 |
| 3 | AI chat prompt | **Done** | §3.5.1 #3 |
| 4 | Live video filter stream | **Done** | §3.5.1 #4 ext. |
| — | Global enhancement module | **Planned** | whole-image tone/color |

---

## License

This project is developed for academic purposes as part of CMPE 491 Senior Design Project at TED University.  
GFPGAN architecture reproduced under [MIT License](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE).
