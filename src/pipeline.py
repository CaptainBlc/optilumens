"""
ImageEnhancementPipeline — orchestrated multi-model enhancement.

CMPE 491 Senior Design Project.

Architecture (post-orchestrator refactor):

    1. analyzeImage()      ImageProfiler  — measure scene characteristics
    2. DecisionEngine      profile        -> ordered list of layers to run
    3. For each chosen layer:
         a) execute the layer
         b) QualityGuard compares before/after, reports trust score
         c) accept / blend / reject the layer's output
    4. computeMetrics() + diffMap() on final image

Available layers (run only when DecisionEngine selects them):
    GFPGAN          face_restorer.py        — face restoration
    REAL_ESRGAN     general_restorer.py     — generic SR/denoise (NEW)
    FACE_PARSE      semantic_parser.py      — 19-class face segmentation
    REGION_ENHANCE  region_enhancer.py      — per-region cosmetic ops
    GLOBAL_ENHANCE  global_enhancer.py      — Furkan's whole-image lift

Team responsibilities:
    Pixel + orchestrator (Batuhan):  GFPGAN, RealESRGAN, decisions, guards
    Face parse + regions:            in-repo FaceParser + RegionEnhancer
    Global Layer (Furkan):           GlobalEnhancer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import time

import numpy as np
import cv2

from profiler import ImageProfiler, ProfileResult
from face_restorer import FaceRestorer, RestorationResult
from general_restorer import GeneralRestorer, GeneralRestorationResult
from metrics import MetricsCalculator, QualityMetrics
from semantic_parser import FaceParser, SemanticResult
from region_enhancer import RegionEnhancer, RegionConfig, RegionEnhanceResult
from global_enhancer import GlobalEnhancer, GlobalEnhancerConfig
from decision_engine import DecisionEngine, Layer, Plan
from quality_guard import QualityGuard, GuardReport


# ── Preset → component-config bridges ─────────────────────────────────
#
# A `ScenePreset` (see scene_presets.py) declares user-intent in coarse,
# named knobs (skin_amount, eyes_sharpen, vibrance, ...).  These two
# helpers translate that intent into the concrete `RegionConfig` and
# `GlobalEnhancerConfig` instances each component already accepts, so
# the chat layer can drive the pipeline without touching its internals.

def _region_cfg_from_preset(preset) -> RegionConfig:
    """Map a ScenePreset onto a RegionConfig (for RegionEnhancer)."""
    return RegionConfig(
        skin_smooth   = float(preset.skin_amount),
        eye_sharpen   = float(preset.eyes_sharpen),
        eye_brighten  = float(preset.eyes_bright),
        lip_vibrance  = float(preset.lips_vibrance),
        lip_warmth    = float(preset.lips_warmth),
        brow_contrast = float(preset.brows_amount),
        nose_sharpen  = float(preset.nose_amount),
    )


def _global_cfg_from_preset(preset) -> GlobalEnhancerConfig:
    """
    Map a ScenePreset onto a GlobalEnhancerConfig.

    Preset fields default to None, in which case we fall back to the
    standard GlobalEnhancer defaults — that way "natural" presets still
    get sensible WB / sharpen even when they only override one knob.
    """
    base = GlobalEnhancerConfig()
    return GlobalEnhancerConfig(
        white_balance  = float(preset.white_balance) if preset.white_balance is not None else base.white_balance,
        shadow_lift    = float(preset.shadow_lift)   if preset.shadow_lift   is not None else base.shadow_lift,
        denoise        = float(preset.bilateral)     if preset.bilateral     is not None else base.denoise,
        clahe_strength = (float(preset.clahe_clip) / 5.0
                          if preset.clahe_clip is not None else base.clahe_strength),
        hdr_tone       = float(preset.hdr)           if preset.hdr           is not None else base.hdr_tone,
        sharpen        = float(preset.sharpen)       if preset.sharpen       is not None else base.sharpen,
        vibrance       = float(preset.vibrance)      if preset.vibrance      is not None else base.vibrance,
        film_look      = float(preset.film_grade)    if preset.film_grade    is not None else base.film_look,
    )


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _wow_dynamic_global_cfg(preset, profile: Optional[ProfileResult]) -> GlobalEnhancerConfig:
    """
    Dynamic WOW mapping:
    - Low-light: lift/denoise up, vibrance/film down (avoid colour drift)
    - Blurry/noisy: sharpen down, denoise up (avoid halos)
    - Clean/bright: allow more punch (vibrance/sharpen up)
    """
    cfg = _global_cfg_from_preset(preset)
    if profile is None:
        return cfg

    b = float(getattr(profile, "brightness", 0.50))
    low = bool(getattr(profile, "is_low_light", False))
    blurry = bool(getattr(profile, "is_blurry", False))
    noisy = bool(getattr(profile, "is_noisy", False))

    # brightness below ~0.55 => low-light strength factor 0..1
    darkness = max(0.0, 0.55 - b) / 0.25
    darkness = max(0.0, min(1.0, darkness))

    # multipliers
    m_lift = 1.0 + 0.35 * darkness if low else 1.0
    m_dn   = 1.0 + (0.40 * darkness if low else 0.0) + (0.25 if noisy else 0.0)
    m_sh   = 1.0 - (0.25 if blurry else 0.0) - (0.10 if noisy else 0.0)
    m_vib  = 1.0 - (0.22 if low else 0.0) - (0.08 if noisy else 0.0)
    m_film = 1.0 - (0.35 if low else 0.0)

    cfg.shadow_lift = _clamp01(cfg.shadow_lift * m_lift)
    cfg.denoise     = _clamp01(cfg.denoise * m_dn)
    cfg.sharpen     = _clamp01(cfg.sharpen * m_sh)
    cfg.vibrance    = _clamp01(cfg.vibrance * m_vib)
    cfg.film_look   = _clamp01(cfg.film_look * m_film)
    return cfg


def _wow_dynamic_region_cfg(preset, profile: Optional[ProfileResult]) -> RegionConfig:
    """
    Dynamic WOW cosmetics:
    - Low-light/noisy: reduce sharpening a bit (avoid crunchy artifacts)
    - Clean: keep strong cosmetics for the wow factor
    """
    cfg = _region_cfg_from_preset(preset)
    if profile is None:
        return cfg
    low = bool(getattr(profile, "is_low_light", False))
    blurry = bool(getattr(profile, "is_blurry", False))
    noisy = bool(getattr(profile, "is_noisy", False))
    if low or blurry or noisy:
        k = 0.85 if (low or noisy) else 0.90
        cfg.eye_sharpen = _clamp01(cfg.eye_sharpen * k)
        cfg.nose_sharpen = _clamp01(cfg.nose_sharpen * k)
        cfg.brow_contrast = _clamp01(cfg.brow_contrast * (0.90 if noisy else 0.95))
    return cfg

def _preset_has_global_overrides(preset) -> bool:
    """True if preset explicitly overrides any global-enhancer knob."""
    if preset is None:
        return False
    return any(getattr(preset, k, None) is not None for k in (
        "white_balance", "shadow_lift", "bilateral", "clahe_clip",
        "hdr", "sharpen", "vibrance", "film_grade",
    ))


def _low_light_natural_global_cfg(profile: Optional[ProfileResult]) -> GlobalEnhancerConfig:
    """
    Mild global configuration for Natural+/Group in low-light.

    Goal: lift shadows + reduce noise without triggering large colour/tone drift
    (which would get rejected by QualityGuard anyway).
    """
    b = float(getattr(profile, "brightness", 0.50))
    # darker scene → slightly stronger lift/denoise; keep colour operations conservative
    darkness = max(0.0, 0.55 - b)
    lift = min(0.70, 0.45 + darkness * 0.90)
    denoise = min(0.55, 0.30 + darkness * 0.70)
    return GlobalEnhancerConfig(
        white_balance=0.55,
        shadow_lift=lift,
        denoise=denoise,
        clahe_strength=0.28,
        hdr_tone=0.22,
        sharpen=0.28,
        vibrance=0.18,
        film_look=0.08,
        temporal_frames=3,
    )


@dataclass
class EnhancementResult:
    """Complete output from one pipeline run."""
    original:       Optional[np.ndarray] = None
    restored:       Optional[np.ndarray] = None     # final image (all layers)
    diff_map:       Optional[np.ndarray] = None
    profile:        Optional[ProfileResult] = None
    metrics:        Optional[QualityMetrics] = None
    restoration:    Optional[RestorationResult] = None
    general_result: Optional[GeneralRestorationResult] = None
    semantic:       Optional[SemanticResult] = None
    region_result:  Optional[RegionEnhanceResult] = None
    plan:           Optional[Plan] = None
    guard_reports:  List[GuardReport] = field(default_factory=list)
    log:            List[str] = field(default_factory=list)
    timings_ms:     Dict[str, float] = field(default_factory=dict)  # per-layer wall time
    success:        bool = False


class ImageEnhancementPipeline:
    """
    Central orchestrator for the GFPGAN-based enhancement system.

    Methods (aligned with Analysis Report class model):
        analyzeImage()  — profile input
        restoreImage()  — full pipeline
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        upscale: int = 1,
        fidelity_weight: float = 0.5,
        region_config: Optional[RegionConfig] = None,
        use_decision_engine: bool = True,
        use_quality_guard: bool = True,
    ) -> None:
        self._profiler = ImageProfiler()
        self._metrics  = MetricsCalculator()

        kw = {}
        if checkpoint_path:
            kw["checkpoint_path"] = checkpoint_path
        self._restorer = FaceRestorer(
            upscale=upscale,
            fidelity_weight=fidelity_weight,
            **kw,
        )

        # General-purpose AI for non-face content
        self._general = GeneralRestorer(scale=2, tile=512, output_scale=0.5)

        # Semantic layer: face parsing + per-region enhancement
        self._parser   = FaceParser()
        self._enhancer = RegionEnhancer(region_config)

        # Global enhancement layer (Furkan)
        self._global_enhancer = GlobalEnhancer()

        # Orchestration helpers
        self._engine = DecisionEngine() if use_decision_engine else None
        self._guard  = QualityGuard()    if use_quality_guard  else None

    # ── public API ────────────────────────────────────────────────

    def analyzeImage(self, image: np.ndarray) -> Optional[ProfileResult]:
        """Evaluate image: lighting, noise, blur, skin presence."""
        return self._profiler.profile(image)

    def restoreImage(
        self,
        image: np.ndarray,
        fidelity_weight: Optional[float] = None,
        only_center_face: bool = False,
        preset=None,
    ) -> EnhancementResult:
        """
        Full pipeline: profile → GFPGAN restore → metrics → diff map.

        Parameters
        ----------
        image : np.ndarray      BGR uint8 input
        fidelity_weight : float 0=max AI, 1=original (override instance default)
        only_center_face : bool Process only the most central face
        preset : ScenePreset    Optional scene-specific override bundle
                                (see scene_presets.py). When given, its
                                fidelity / force_* / skip_* fields override
                                the slider value and the DecisionEngine plan
                                so the same pipeline produces a recognisable
                                look per scenario (portrait, old_photo,
                                magazine, low_light, ...).
        """
        if preset is not None and fidelity_weight is None:
            fidelity_weight = float(preset.fidelity)
        result = EnhancementResult(original=image.copy() if image is not None else None)
        log = result.log
        timings = result.timings_ms

        def _timeit(key: str, fn):
            t0 = time.perf_counter()
            out = fn()
            timings[key] = (time.perf_counter() - t0) * 1000.0
            return out

        if image is None or image.size == 0:
            log.append("[ERROR] Invalid input")
            return result

        log.append("=== ImageEnhancementPipeline ===")
        log.append("Input: {}x{} ({:.1f} MP)".format(
            image.shape[1], image.shape[0],
            image.shape[0] * image.shape[1] / 1e6))

        # ── Step 1: analyzeImage() ──
        profile = _timeit("analyze", lambda: self.analyzeImage(image))
        result.profile = profile
        if profile:
            log.append("--- analyzeImage() ---")
            log.append("  brightness   = {:.3f} {}".format(
                profile.brightness,
                "(LOW)" if profile.is_low_light else
                "(HIGH)" if profile.is_overexposed else ""))
            log.append("  contrast     = {:.3f}".format(profile.contrast))
            log.append("  blur_score   = {:.1f} {}".format(
                profile.blur_score,
                "(BLURRY)" if profile.is_blurry else "(SHARP)"))
            log.append("  noise_level  = {:.1f} {}".format(
                profile.noise_level,
                "(NOISY)" if profile.is_noisy else ""))
            log.append("  skin_ratio   = {:.3f} {}".format(
                profile.skin_ratio,
                "(FACE DETECTED)" if profile.has_skin else ""))

        # ── Step 2: DecisionEngine — decide which layers to execute ──
        # Probe face count cheaply via the profile's skin ratio (engine
        # treats skin > 4% as a positive face indicator).  We don't run
        # the heavy face detector twice — the FaceRestorer call below
        # will do its own detection if GFPGAN is in the plan.
        if self._engine is not None:
            plan = _timeit(
                "decide",
                lambda: self._engine.decide(
                    profile,
                    faces_found=0,
                    image_shape=image.shape,
                    preset_name=str(getattr(preset, "name", "")) if preset is not None else None,
                ),
            )
            result.plan = plan
            log.append("--- DecisionEngine ---")
            log.extend(plan.justification())
        else:
            plan = None

        current = image.copy()  # this is the "rolling" image as layers run

        # Preset overrides on the plan: a preset can force a layer on/off.
        # We do this *after* the engine has explained itself, so the log
        # still shows the engine's reasoning, and then the preset's
        # override is recorded as an extra line.
        if plan is not None and preset is not None:
            for d in plan.steps:
                # Presentation tuning: WOW should look strong but must not feel broken/laggy.
                # On CPU, GFPGAN + Real-ESRGAN together can easily exceed 4s. We keep WOW's
                # punch primarily via Region+Global, and only run heavy AI when it is likely
                # to materially improve quality.
                pn = str(getattr(preset, "name", "") or "").lower()
                if pn == "wow" and profile is not None:
                    # GFPGAN: run only when the face really needs restoration (very blurry or low-light).
                    # Otherwise skip to avoid a ~3s stall.
                    if d.layer == Layer.GFPGAN and d.run:
                        very_blur = float(getattr(profile, "blur_score", 0.0)) < 60.0
                        low_light = bool(getattr(profile, "is_low_light", False))
                        if not (very_blur or low_light):
                            d.run = False
                            log.append("  [WOW] GFPGAN suppressed (perf) — not very blurry / low-light")

                    # Real-ESRGAN: run when needed (very blurry, low-light, noisy, or very low contrast).
                    if d.layer == Layer.REAL_ESRGAN and d.run:
                        very_blur = bool(getattr(profile, "is_blurry", False)) and float(getattr(profile, "blur_score", 0.0)) < 60.0
                        low_light = bool(getattr(profile, "is_low_light", False))
                        noisy = bool(getattr(profile, "is_noisy", False))
                        low_contrast = float(getattr(profile, "contrast", 1.0)) < 0.18
                        if not (very_blur or low_light or noisy or low_contrast):
                            d.run = False
                            log.append("  [WOW] REAL_ESRGAN suppressed (perf) — frame is clean enough")

                if d.layer == Layer.REAL_ESRGAN:
                    if preset.skip_general and d.run:
                        d.run = False
                        log.append("  [PRESET] REAL_ESRGAN forced OFF "
                                   "by '{}' preset".format(preset.name))
                    elif preset.force_general and not d.run:
                        # Even when a preset asks for SR/denoise, avoid forcing it on
                        # perfectly clean frames: Real-ESRGAN is the slowest CPU layer
                        # after GFPGAN and can add 1s+ latency at VGA resolutions.
                        needs_general = bool(
                            getattr(profile, "is_low_light", False)
                            or getattr(profile, "is_blurry", False)
                            or getattr(profile, "is_noisy", False)
                            or (profile is not None and float(getattr(profile, "contrast", 1.0)) < 0.18)
                        )
                        if needs_general:
                            d.run = True
                            log.append("  [PRESET] REAL_ESRGAN forced ON "
                                       "by '{}' preset (needs cleanup)".format(preset.name))
                        else:
                            log.append("  [PRESET] REAL_ESRGAN force suppressed "
                                       "on clean frame (perf) by '{}' preset".format(preset.name))
                if d.layer == Layer.GLOBAL_ENHANCE:
                    if preset.skip_global and d.run:
                        d.run = False
                        log.append("  [PRESET] GLOBAL_ENHANCE forced OFF "
                                   "by '{}' preset".format(preset.name))
                    elif preset.force_global and not d.run:
                        d.run = True
                        log.append("  [PRESET] GLOBAL_ENHANCE forced ON "
                                   "by '{}' preset".format(preset.name))

        def _layer_should_run(layer: Layer) -> bool:
            if plan is None:
                return True   # all-on if engine disabled
            for d in plan.steps:
                if d.layer == layer:
                    return d.run
            return False

        def _apply_guard(before, after, label, dst_attr):
            """Run guard, append report, return image to use downstream."""
            if self._guard is None or after is None:
                return after
            # Preset-specific guardrails: for "natural" looks we tighten
            # thresholds and blend more aggressively to avoid over-processing.
            guard = self._guard
            if preset is not None:
                pn = str(getattr(preset, "name", "") or "").lower()
                if pn in ("natural", "natural_plus", "group"):
                    if label == "GlobalEnhance":
                        # Global passes can drift colour/tone; prefer blending over hard reject
                        guard = QualityGuard(accept_threshold=74.0, warn_threshold=50.0, blend_ratio=0.65)
                    elif label == "RegionEnhance":
                        guard = QualityGuard(accept_threshold=75.0, warn_threshold=60.0, blend_ratio=0.55)
                    elif label == "GFPGAN":
                        guard = QualityGuard(accept_threshold=72.0, warn_threshold=55.0, blend_ratio=0.50)
                    elif label == "RealESRGAN":
                        guard = QualityGuard(accept_threshold=72.0, warn_threshold=55.0, blend_ratio=0.45)

            rep = guard.evaluate(before, after, label=label)
            result.guard_reports.append(rep)
            log.append(rep.log[0])
            return rep.output

        # ── Step 3: GFPGAN — face restoration ──
        if _layer_should_run(Layer.GFPGAN):
            log.append("--- GFPGAN Restoration ---")
            resto = _timeit(
                "gfpgan",
                lambda: self._restorer.restore(
                    current,
                    fidelity_weight=fidelity_weight,
                    only_center_face=only_center_face,
                ),
            )
            result.restoration = resto
            log.extend(resto.log)
            log.append("  Faces found : {}".format(resto.faces_found))
            if resto.success and resto.restored is not None:
                current = _apply_guard(
                    current, resto.restored, "GFPGAN", "restoration")

        # ── Step 4: Real-ESRGAN — general AI for non-face / low-quality ──
        if _layer_should_run(Layer.REAL_ESRGAN):
            log.append("--- Real-ESRGAN ---")
            gr = _timeit("realesrgan", lambda: self._general.restore(current))
            result.general_result = gr
            log.extend(gr.log)
            if gr.success and gr.restored is not None:
                current = _apply_guard(
                    current, gr.restored, "RealESRGAN", "general_result")

        # ── Step 5: Face parse + region cosmetic ──
        if _layer_should_run(Layer.FACE_PARSE):
            log.append("--- Semantic Layer ---")
            pn_preset = str(getattr(preset, "name", "") or "").lower() if preset is not None else ""
            # Performance: face parsing (RetinaFace + BiSeNet) can be very slow on CPU.
            # For "wow" we accept doing cosmetics on a smaller proxy image and upscaling
            # the final cosmetic result back; Guard still protects the output.
            fast_cosmetic = pn_preset == "wow"
            base_img = current
            if fast_cosmetic:
                try:
                    h0, w0 = base_img.shape[:2]
                    target = 320
                    if max(h0, w0) > target:
                        s = target / float(max(h0, w0))
                        base_img = cv2.resize(
                            base_img,
                            (max(1, int(w0 * s)), max(1, int(h0 * s))),
                            interpolation=cv2.INTER_AREA,
                        )
                        log.append("  [WOW] fast cosmetic proxy: {}x{} -> {}x{}".format(
                            w0, h0, base_img.shape[1], base_img.shape[0]))
                except Exception:
                    base_img = current

            sem = _timeit("face_parse", lambda: self._parser.parse(base_img))
            result.semantic = sem
            log.extend("  " + ln for ln in sem.log)

            if sem.success and sem.faces and _layer_should_run(Layer.REGION_ENHANCE):
                # Preset overrides the per-region knobs; without one
                # we use the pipeline's default RegionEnhancer so user
                # preferences from __init__ are preserved.
                if preset is not None:
                    pn_cfg = str(getattr(preset, "name", "") or "").lower()
                    if pn_cfg == "wow":
                        reg_cfg = _wow_dynamic_region_cfg(preset, profile)
                        reg_enh = RegionEnhancer(reg_cfg)
                        log.append("  [WOW] dynamic region cfg applied")
                    else:
                        reg_enh = RegionEnhancer(_region_cfg_from_preset(preset))
                    log.append("  [PRESET] region knobs from '{}' "
                               "(skin={:.2f} eyes={:.2f} lips={:.2f})".format(
                                   preset.name, preset.skin_amount,
                                   preset.eyes_sharpen, preset.lips_vibrance))
                else:
                    reg_enh = self._enhancer
                reg = _timeit("region", lambda: reg_enh.apply(base_img, sem))
                result.region_result = reg
                log.extend("  " + ln for ln in reg.log)
                if reg.success and reg.image is not None:
                    out_img = reg.image
                    if fast_cosmetic and out_img.shape[:2] != current.shape[:2]:
                        try:
                            out_img = cv2.resize(
                                out_img, (current.shape[1], current.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        except Exception:
                            out_img = reg.image
                    current = _apply_guard(
                        current, out_img, "RegionEnhance", "region_result")

        # ── Step 6: Global Enhancement (Furkan) ──
        if _layer_should_run(Layer.GLOBAL_ENHANCE):
            log.append("--- GlobalEnhancer ---")
            try:
                # Preset overrides the global knobs (WB, vibrance,
                # sharpen, …). When absent, use the shared instance so
                # the temporal frame buffer stays consistent.
                pn = str(getattr(preset, "name", "") or "").lower() if preset is not None else ""
                if pn in ("natural_plus", "group") and (
                    getattr(profile, "is_low_light", False) or getattr(profile, "is_blurry", False)
                ):
                    glob_enh = GlobalEnhancer(_low_light_natural_global_cfg(profile))
                    tag = "low-light" if getattr(profile, "is_low_light", False) else "blurry"
                    log.append("  [PRESET] mild global ({}) for '{}'".format(tag, pn))
                elif preset is not None and _preset_has_global_overrides(preset):
                    pn2 = str(getattr(preset, "name", "") or "").lower()
                    if pn2 == "wow":
                        glob_enh = GlobalEnhancer(_wow_dynamic_global_cfg(preset, profile))
                        log.append("  [WOW] dynamic global cfg applied")
                    else:
                        glob_enh = GlobalEnhancer(_global_cfg_from_preset(preset))
                        log.append("  [PRESET] global overrides from '{}'".format(
                            preset.name))
                else:
                    glob_enh = self._global_enhancer
                ge = _timeit(
                    "global",
                    lambda: glob_enh.enhance(current, profile=profile, use_temporal=False),
                )
                if ge.success and ge.image is not None:
                    log.extend("  " + ln for ln in ge.log)
                    current = _apply_guard(
                        current, ge.image, "GlobalEnhance", None)
                else:
                    log.append("  GlobalEnhancer returned no result -- passthrough")
            except Exception as _e:
                log.append("  [WARN] GlobalEnhancer failed: {}".format(_e))

        result.restored = current

        # Step 6: Quality Metrics
        log.append("--- Quality Metrics ---")
        try:
            m = _timeit("metrics", lambda: self._metrics.compute_all(image, result.restored))
            result.metrics = m
            log.append("  PSNR         = {:.2f} dB".format(m.psnr))
            log.append("  SSIM         = {:.4f}".format(m.ssim))
            log.append("  Entropy      = {:.3f} → {:.3f} (Δ{:+.3f})".format(
                m.entropy_before, m.entropy_after,
                m.entropy_after - m.entropy_before))
            log.append("  Colorfulness = {:.2f} → {:.2f} (Δ{:+.2f})".format(
                m.colorfulness_before, m.colorfulness_after,
                m.colorfulness_after - m.colorfulness_before))
        except Exception as e:
            log.append("[WARN] Metrics failed: {}".format(e))

        # Step 7: Difference map
        try:
            result.diff_map = _timeit(
                "diff_map", lambda: self._metrics.difference_map(image, result.restored)
            )
            log.append("  Difference map: generated")
        except Exception as e:
            log.append("[WARN] Diff map failed: {}".format(e))

        result.success = True
        if timings:
            total = sum(timings.values())
            keys = ["analyze", "decide", "gfpgan", "realesrgan", "face_parse", "region", "global", "metrics", "diff_map"]
            shown = ["{}={:.0f}ms".format(k, timings[k]) for k in keys if k in timings and timings[k] >= 1.0]
            log.append("--- Timing ---")
            log.append("  Total: {:.0f} ms".format(total))
            if shown:
                log.append("  " + "  ".join(shown))
        log.append("=== Pipeline completed ===")
        return result
