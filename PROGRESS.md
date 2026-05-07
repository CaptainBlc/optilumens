# PROGRESS.md — OptiLumen Development Log

> **Shared AI memory across machines.** Update this after every meaningful change.
> Any Cursor/Claude session on any PC should read this file FIRST to understand the current state.
> Structured so both humans and AI can parse it quickly.

**Repo:** `https://github.com/CaptainBlc/Global-Enhancement`
**Active branch:** `Batuhan-Develop`
**Owner (this machine):** Batuhan Taşdemir — Pixel Layer + Orchestration

---

## Current Status

**Phase:** Orchestrator refactor — Decision Engine + Quality Guard + Real-ESRGAN added on `Batuhan-Develop`. Pipeline is now profile-driven and audit-ready.
**Active Sprint:** Defense-readiness pass.
**Blocked on:** nothing
**Safe rollback point:** tag `v1.0-pre-orchestrator` on `main` (pushed to remote).

### What's new in this branch (vs `main`)
- `src/decision_engine.py` — profile-based layer selector (RUN/SKIP justification per layer)
- `src/quality_guard.py` — anti-hallucination SSIM guard with accept/blend/reject
- `src/general_restorer.py` — Real-ESRGAN x2plus wrapper, lazy imports, tile inference
- `src/models/rrdb_arch.py` — pure-PyTorch RRDBNet backbone (no `basicsr`)
- `src/pipeline.py` — orchestrated multi-model run; only chosen layers execute, every AI step is guarded
- `src/face_restorer.py` — lenient state-dict load + GFPGAN ResBlock fix (down/up modes, conv ordering)
- `scripts/download_model.py` — multi-model fetcher (`--only gfpgan|realesrgan`)
- `src/gui/main_window.py` — info panel now shows decision plan + per-layer trust scores

### Known issues
- GFPGAN encoder/decoder load 230/285 weights; the SFT condition heads and StyleGAN2 toRGB tail need a deeper architecture pass to fully load the official `GFPGANv1.3.pth`. System gracefully falls back to classical face sharpening when the StyleGAN2 forward fails. **Pipeline still works end-to-end** because Real-ESRGAN, FaceParser, RegionEnhancer, and GlobalEnhancer cover the same use case.

---

## Sprint Roadmap

Derived from Analysis Report §3.5.1 use-case scenarios.

| # | Name | Scenario | Status |
|---|------|----------|--------|
| 1 | Real-time camera capture | §3.5.1 #4 | **Done** (commit `eac998b`) |
| 2 | Background batch rendering | §3.5.1 #1, #2 | **Done** (commit `87f366d`) |
| 3 | AI chat prompt for photo editing | §3.5.1 #3 | **Done** |
| 4 | Live video filter stream | §3.5.1 #4 ext. | **Done** |

---

## Changelog

### 2026-05-07 — Orchestrator refactor on `Batuhan-Develop`
- **New `src/decision_engine.py`** — `DecisionEngine.decide(profile, faces_found, image_shape)` returns a `Plan` of typed `Decision`s. Each layer is justified individually so the pipeline log explicitly states *why* GFPGAN ran or didn't, why Real-ESRGAN was skipped, etc. (HLD §1 explainable processing requirement).
- **New `src/quality_guard.py`** — model-agnostic anti-hallucination guard. Computes SSIM + pixel drift between any layer's input/output, converts to a 0–100 trust score, and decides accept (≥70) / blend (≥50) / reject (<50). Caught GlobalEnhancer over-cooking on a sharp face image during testing — blended back 40 % to original.
- **New `src/general_restorer.py` + `src/models/rrdb_arch.py`** — Real-ESRGAN x2plus integration written in pure PyTorch (no `basicsr` dependency, same approach we used for GFPGAN). Lazy `torch` import, tile-based inference for big frames, conservative classical fallback if weights aren't downloaded yet.
- **`src/face_restorer.py`** — switched to lenient state-dict load. The reproduced GFPGAN architecture had two bugs found while debugging: (1) `ResBlock` had `conv1`/`conv2` swapped in channel-transition role, (2) `ResBlock` was missing the down/up interpolation. Both fixed in `src/models/gfpgan_arch.py`. Encoder + decoder now load ~230/285 official weights cleanly. SFT/toRGB tail still needs work; system gracefully degrades to classical when the GAN forward fails.
- **`src/pipeline.py`** — `restoreImage()` now: profile → DecisionEngine → execute only chosen layers → guard each AI step → metrics + diff. Returns `EnhancementResult` with new `plan` and `guard_reports` fields.
- **`src/gui/main_window.py`** — `_on_done` now appends a compact "Layers run" + "Trust scores" block to the info panel after every Restore.
- **Docs:** `PROGRESS.md` updated with this branch's diff and known issues.
- **Safety:** `git tag v1.0-pre-orchestrator` pushed before any of these changes.

### 2026-05-06 — Merge `Furkan-Develop` into `main` (semantic + region path)
- **Merged from** `origin/Furkan-Develop`: `src/semantic_parser.py` (`FaceParser`), `src/region_enhancer.py` (`RegionEnhancer`), `src/live_capture.py`, and `src/pipeline.py` wiring so `restoreImage()` runs GFPGAN then optional parse + per-region enhancement when faces parse successfully.
- **Docs:** `README.md` conflict resolved (unified layer diagram + accurate class table). `BRANCH_INFO.md` updated upstream.
- **Note:** Global whole-image enhancer is still the logged placeholder; GUI and Pixel sprints (batch, camera, chat, live filters) unchanged.

### 2026-04-22 — Sprint 4: Live Video Filter Stream (Scenario #4 ext.)
- **New:** `src/live_filters.py` with three filters sharing a `BaseFilter.apply(frame) → frame` contract:
  - `NoFilter` (OFF) — identity, <1 ms.
  - `BeautyFilter` (BEAUTY) — bilateral skin-smoothing + CLAHE on L + unsharp + warm-tone LUT. ~40–130 ms on 720p; real-time for demo.
  - `AIFilter` (AI) — best-effort GFPGAN wrapper: downscales to 512 px, processes every 10th frame, caches output. Falls back to BeautyFilter whenever torch/weights are absent.
- **GUI:** filter button group (`OFF / BEAUTY / AI`) lives in the toolbar, visible only while Live mode is active. Preview frames run through the active filter before display, and the filter is applied to snapshots so the captured still matches what the user saw.
- **Design note:** CPU GFPGAN is fundamentally too slow for real-time video; `AIFilter` is an honest hybrid — classical look every frame, AI pass cached every Nth. A CUDA GPU or a lighter restoration model is the proper upgrade path.

### 2026-04-22 — Sprint 3: AI Chat Prompt (Scenario #3)
- **New:** `src/chat_commands.py` — rule-based intent parser (Turkish + English) with 12 intents (`restore`, `set_fidelity`, `adjust_fidelity`, `center_only`, `view`, `reset`, `save`, `open`, `live`, `capture`, `batch`, `help`) + an `unknown` fallback with a friendly hint. `Command` dataclass + `ChatExecutor.dispatch()` glue wire commands to `MainWindow` methods.
- **GUI:** chat bar at the bottom of the main window (red accent). Hitting Enter dispatches the command, appends a conversation entry to the main log (user → intent / params / result).
- **Design choice:** MVP is rule-based — offline, zero API cost, fully explainable (important for a senior design demo). The `Command` protocol is LLM-compatible; swapping `parse()` for an LLM adapter is the only change needed to upgrade.

### 2026-04-22 — Sprint 2: Background Batch Rendering (Scenarios #1, #2)
- **New:** `src/batch_processor.py` — `BatchWorker(QThread)` processes a list of files through `ImageEnhancementPipeline`. Supports cancel, emits `progress / log_line / finished_all` signals. `collect_images(folder)` scans for `.jpg .jpeg .png .bmp .tiff .webp`.
- **GUI:** `BatchDialog` (non-modal) shows a progress bar, per-file status table (index / file / status / time), live log line, and Cancel / Close / Open-Output buttons. New `Batch…` toolbar button wires a folder picker to the dialog. Outputs written to `outputs/batch_<timestamp>/`.
- **Why:** GFPGAN inference on CPU is slow enough that the main window would freeze on multi-image folders. Spinning up a QThread keeps the UI responsive and lets users see per-file progress.

### 2026-04-22 — Sprint 1: Real-Time Camera (Scenario 4)
- **New:** `src/camera_capture.py` — `CameraCapture` wrapper around OpenCV.
  - DSHOW backend on Windows, device enumeration, stabilised snapshot helper.
- **GUI:** Toolbar `● Live` toggle + `Capture` button, ~30 FPS preview via `QTimer`, snapshots flow into the existing pipeline as if loaded from disk.
- **Docs:** `README.md` Live Camera section + Roadmap table.
- **Commit:** `eac998b` on `main`, fast-forwarded to `Batuhan-Develop`, merged into `Total-Develop`.

### 2026-04-22 — Multi-PC handoff system
- **New:** `.cursor/rules/optilumen.mdc` — project context rules that Cursor AI reads automatically.
- **New:** `PROGRESS.md` (this file) — structured, always-on dev log.
- **Docs:** README "Continuing on another PC" section.

### Earlier (pre-summary) — see commit history
- 2026-04-22 Refactor to GFPGAN v1.3 architecture (StyleGAN2 + SFT, pure PyTorch).
- 2026-04-22 6-branch GitHub layout established (`main`, `legacy`, `*-Develop`, `Total-Develop`).
- 2026-04-22 Classical pixel system archived to `legacy/`.

---

## Architecture Snapshot

```
Input ─► Semantic/Edge (Emir, external)
      ─► Pixel Layer (us) ─► GFPGAN v1.3 face restoration
      ─► Global Enhancement (Furkan, external)
      ─► Fusion → Output
```

**Pixel Layer public API:**
- `ImageEnhancementPipeline.restoreImage(img) → EnhancementResult`
- `FaceRestorer.restore(img) → RestorationResult`
- `CameraCapture.snapshot() → np.ndarray`  *(Sprint 1)*
- `BatchWorker(files, out_dir).start()` + signals  *(Sprint 2)*
- `chat_commands.parse(text) → Command` + `ChatExecutor.dispatch(cmd, window)` *(Sprint 3)*
- `live_filters.make_filter(name).apply(frame) → frame` — OFF/BEAUTY/AI  *(Sprint 4)*

---

## Open Decisions / Questions

*(Update when a question surfaces that needs team discussion.)*

- [ ] Sprint 2: Do we allow recursive folder scanning in batch mode? → **Decision:** Yes, single-level first; add `--recursive` flag later.
- [x] Sprint 3: Which LLM for chat prompt? → **Decision:** start rule-based for MVP (offline + demo-safe). Swap in an LLM later via a parser plug-in if demo time allows.
- [ ] Sprint 4: Max live FPS acceptable with GFPGAN? → Need benchmark; GFPGAN on CPU is ~2-3 s/face, so live filter will need a lighter model or skip frames.

---

## How to Resume on Another PC

1. Install prerequisites: **Git**, **Python 3.10-3.13**, **Cursor IDE**.
2. Clone and bootstrap:
   ```
   git clone https://github.com/CaptainBlc/Global-Enhancement.git
   cd Global-Enhancement
   py -3 -m venv venv
   venv\Scripts\activate
   python setup.py           # installs deps + downloads GFPGANv1.3.pth
   ```
3. Open the folder in Cursor. The AI will auto-read `.cursor/rules/optilumen.mdc` and this file — no manual context-setting required.
4. Pull first before any edit: `git pull --ff-only`.
5. After your changes, update **this file's Changelog** and push. That's the handoff.

---

## Next Actions (queue)

All four roadmap sprints delivered. Remaining ideas, ordered by presentation impact:

1. **Integration milestone** — connect Furkan's Global Enhancement Layer and Emir's Semantic/Edge Layer once their `main` branches are ready. Pipeline wrapper exists; we only need to populate two call sites.
2. **Performance** — GPU path (CUDA torch wheel auto-detect) for real-time AI filter. Lighter model (GPEN, CodeFormer-tiny) as an alternative.
3. **Stretch**: swap rule-based chat parser for a local Phi-3 LLM adapter.
4. **Quality of life**: recursive batch scan, per-run JSON report, in-GUI side-by-side compare of multiple batch results.
