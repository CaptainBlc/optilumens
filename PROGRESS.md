# PROGRESS.md — OptiLumen Development Log

> **Shared AI memory across machines.** Update this after every meaningful change.
> Any Cursor/Claude session on any PC should read this file FIRST to understand the current state.
> Structured so both humans and AI can parse it quickly.

**Repo:** `https://github.com/CaptainBlc/Global-Enhancement`
**Active branch:** `main`
**Owner (this machine):** Batuhan Taşdemir — Pixel Layer

---

## Current Status

**Phase:** Post-refactor → GFPGAN-based system is live, GUI works, model auto-downloads.
**Active Sprint:** Sprint 4 — Live video filter (planning)
**Blocked on:** nothing

---

## Sprint Roadmap

Derived from Analysis Report §3.5.1 use-case scenarios.

| # | Name | Scenario | Status |
|---|------|----------|--------|
| 1 | Real-time camera capture | §3.5.1 #4 | **Done** (commit `eac998b`) |
| 2 | Background batch rendering | §3.5.1 #1, #2 | **Done** (commit `87f366d`) |
| 3 | AI chat prompt for photo editing | §3.5.1 #3 | **Done** |
| 4 | Live video filter stream | §3.5.1 #4 ext. | Planned |

---

## Changelog

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

1. **Sprint 4:** Live-video filter — process preview frames at downscaled res + skip frames; evaluate whether GFPGAN is viable real-time or if a lighter face-restoration model is needed.
2. (Backlog, Sprint 3+) Swap rule-based parser for an LLM adapter (local Phi-3 preferred; OpenAI API as fallback).
3. (Backlog, Sprint 2+) Batch mode: recursive folder scan, JSON results report, parallel workers on multi-core CPUs.
