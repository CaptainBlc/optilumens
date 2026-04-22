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
**Active Sprint:** Sprint 2 — Background batch rendering
**Blocked on:** nothing

---

## Sprint Roadmap

Derived from Analysis Report §3.5.1 use-case scenarios.

| # | Name | Scenario | Status |
|---|------|----------|--------|
| 1 | Real-time camera capture | §3.5.1 #4 | **Done** (commit `eac998b`) |
| 2 | Background batch rendering | §3.5.1 #1, #2 | **In progress** |
| 3 | AI chat prompt for photo editing | §3.5.1 #3 | Planned |
| 4 | Live video filter stream | §3.5.1 #4 ext. | Planned |

---

## Changelog

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
- `ImageEnhancementPipeline.enhance(img) → EnhancementResult`
- `FaceRestorer.restore(img) → RestorationResult`
- `CameraCapture.snapshot() → np.ndarray`  *(Sprint 1)*

---

## Open Decisions / Questions

*(Update when a question surfaces that needs team discussion.)*

- [ ] Sprint 2: Do we allow recursive folder scanning in batch mode? → **Decision:** Yes, single-level first; add `--recursive` flag later.
- [ ] Sprint 3: Which LLM for chat prompt? Local (Phi-3) vs API (OpenAI)? → Open.
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

1. **Sprint 2:** Batch folder processing — `QThread` worker, progress panel with per-file status + cancel, results summary.
2. **Sprint 3:** Chat prompt UX mockup → pick LLM backend → integrate.
3. **Sprint 4:** Live-video filter — downscale + skip frames; consider a lighter model for preview path.
