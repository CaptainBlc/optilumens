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

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _auto_strength(profile: Optional[ProfileResult]) -> float:
    """Overall enhancement strength in [0..1] from profile signals."""
    if profile is None:
        return 0.55
    b = float(getattr(profile, "brightness", 0.50))
    c = float(getattr(profile, "contrast", 0.25))
    blur = float(getattr(profile, "blur_score", 120.0))
    noise = float(getattr(profile, "noise_level", 6.0))
    low = bool(getattr(profile, "is_low_light", False))
    ov = bool(getattr(profile, "is_overexposed", False))

    blur_need = _clamp((130.0 - blur) / 90.0, 0.0, 1.0)          # <40 => high need
    noise_need = _clamp((noise - 2.5) / 10.0, 0.0, 1.0)
    contrast_need = _clamp((0.28 - c) / 0.18, 0.0, 1.0)
    light_need = _clamp((0.52 - b) / 0.22, 0.0, 1.0) if low else 0.0

    s = (
        0.40 * blur_need +
        0.25 * contrast_need +
        0.20 * noise_need +
        0.15 * light_need
    )
    if ov:
        s = max(0.0, s - 0.35)
    return _clamp01(s)


def _scale_region_cfg(cfg: RegionConfig, k: float) -> RegionConfig:
    """Scale cosmetic knobs safely. k in [0..1.3]."""
    kk = _clamp(k, 0.0, 1.30)
    return RegionConfig(
        skin_smooth=_clamp01(cfg.skin_smooth * kk),
        eye_sharpen=_clamp01(cfg.eye_sharpen * kk),
        eye_brighten=_clamp01(cfg.eye_brighten * kk),
        lip_vibrance=_clamp01(cfg.lip_vibrance * kk),
        lip_warmth=_clamp01(cfg.lip_warmth * kk),
        brow_contrast=_clamp01(cfg.brow_contrast * kk),
        nose_sharpen=_clamp01(cfg.nose_sharpen * kk),
    )


def _scale_global_cfg(cfg: GlobalEnhancerConfig, k: float) -> GlobalEnhancerConfig:
    """Scale global knobs safely. k in [0..1.3]."""
    kk = _clamp(k, 0.0, 1.30)
    return GlobalEnhancerConfig(
        white_balance=_clamp01(cfg.white_balance * kk),
        shadow_lift=_clamp01(cfg.shadow_lift * kk),
        denoise=_clamp01(cfg.denoise * kk),
        clahe_strength=_clamp01(cfg.clahe_strength * kk),
        hdr_tone=_clamp01(cfg.hdr_tone * kk),
        sharpen=_clamp01(cfg.sharpen * kk),
        vibrance=_clamp01(cfg.vibrance * kk),
        film_look=_clamp01(cfg.film_look * kk),
        temporal_frames=int(getattr(cfg, "temporal_frames", 3) or 3),
    )


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
    c = float(getattr(profile, "contrast", 0.25))

    # brightness below ~0.55 => low-light strength factor 0..1
    darkness = max(0.0, 0.55 - b) / 0.25
    darkness = max(0.0, min(1.0, darkness))

    # multipliers
    m_lift = 1.0 + 0.35 * darkness if low else 1.0
    m_dn   = 1.0 + (0.40 * darkness if low else 0.0) + (0.25 if noisy else 0.0)
    m_sh   = 1.0 - (0.25 if blurry else 0.0) - (0.10 if noisy else 0.0)
    m_vib  = 1.0 - (0.22 if low else 0.0) - (0.08 if noisy else 0.0)
    m_film = 1.0 - (0.35 if low else 0.0)

    # Bright / already-contrasty scenes: reduce local contrast and colour punch
    # to avoid "plastic" oversaturation and harsh tone mapping.
    if b >= 0.60:
        m_vib *= 0.85
        m_film *= 0.85
    if c >= 0.28:
        cfg.clahe_strength = _clamp01(cfg.clahe_strength * 0.80)
        cfg.hdr_tone = _clamp01(cfg.hdr_tone * 0.85)

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

        mp_in = (image.shape[0] * image.shape[1]) / 1e6
        preset_name = str(getattr(preset, "name", "") or "").lower() if preset is not None else ""

        def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
            h, w = img.shape[:2]
            m = max(h, w)
            if m <= max_side:
                return img
            s = max_side / float(m)
            return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)

        def _highlight_ratio(img: np.ndarray) -> float:
            """Fraction of very bright pixels (backlit indicator)."""
            try:
                small = _resize_max_side(img, 320)
                g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                return float(np.mean(g >= 245))
            except Exception:
                return 0.0

        def _face_mask_from_semantic(sem) -> Optional[np.ndarray]:
            """Return a soft float32 [0..1] mask for the primary face region."""
            if sem is None or not getattr(sem, "faces", None):
                return None
            try:
                face0 = sem.faces[0]
                skin = None
                try:
                    skin = face0.regions.get("skin")
                except Exception:
                    skin = None
                if skin is not None and getattr(skin, "mask", None) is not None:
                    m = skin.mask.astype(np.float32) / 255.0
                elif getattr(sem, "label_map", None) is not None:
                    m = (sem.label_map > 0).astype(np.float32)
                else:
                    return None
                if m.shape[:2] != (image.shape[0], image.shape[1]):
                    m = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                m = cv2.GaussianBlur(m, (0, 0), 9.0)
                return np.clip(m, 0.0, 1.0)
            except Exception:
                return None

        def _apply_face_micro_polish(img_in: np.ndarray, sem, amount: float) -> Optional[np.ndarray]:
            """Very mild face-only micro sharpening (safe phone-like crispness)."""
            m = _face_mask_from_semantic(sem)
            if m is None:
                return None
            try:
                a = float(np.clip(amount, 0.0, 1.0))
                if a <= 0.01:
                    return None
                img_f = img_in.astype(np.float32)
                blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=1.0)
                sharp = img_f + (0.60 * a) * (img_f - blur)
                sharp = np.clip(sharp, 0, 255).astype(np.uint8)
                out = (
                    img_in.astype(np.float32) * (1.0 - m[..., None]) +
                    sharp.astype(np.float32) * m[..., None]
                ).astype(np.uint8)
                return out
            except Exception:
                return None

        # Performance policy for very large images:
        # - profiling/metrics/global are O(HW). Run them on a proxy to avoid multi-second stalls.
        # - AI layers still run as decided (they operate on faces/tiles), but we avoid full-res
        #   classical stacks unless explicitly needed.
        perf_large = mp_in >= 4.0
        prof_img = image
        if perf_large:
            prof_img = _resize_max_side(image, 1024)
            log.append("  [PERF] large input {:.1f}MP → proxy {}x{} for analyze/metrics/global".format(
                mp_in, prof_img.shape[1], prof_img.shape[0]
            ))

        # ── Step 1: analyzeImage() ──
        profile = _timeit("analyze", lambda: self.analyzeImage(prof_img))
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

        # ── Auto dynamics: continuous strength scaling ──
        strength = _auto_strength(profile)
        # Map [0..1] => [0.80..1.20] (clean -> milder, poor -> stronger)
        k_auto = 0.80 + 0.40 * strength
        log.append("  [AUTO] strength={:.2f}  scale={:.2f}".format(strength, k_auto))
        if fidelity_weight is None:
            # Poor frames -> lower fw (more AI). Clean frames -> higher fw (more original).
            fidelity_weight = _clamp(0.75 - 0.35 * strength, 0.30, 0.85)

        # ── Step 2: DecisionEngine — decide which layers to execute ──
        # IMPORTANT: skin_ratio can be misleading on webcam/backlit scenes.
        # Use a fast RetinaFace-only probe to get a reliable faces_found count
        # for the DecisionEngine without running the full parser or GFPGAN.
        if self._engine is not None:
            faces_probe = 0
            try:
                # Detection on a modest proxy keeps it fast and stable.
                det_img = _resize_max_side(image, 640)
                faces_probe = int(getattr(self._parser, "detect_faces_count", lambda _x: 0)(det_img))
            except Exception:
                faces_probe = 0

            # ── Preset: automatic selection (default) ──
            # If the user didn't specify a preset (GUI default), choose one based on
            # the profile + detected face count. Users can still override via chat.
            if preset is None and profile is not None:
                try:
                    from scene_presets import PRESETS
                    pn_auto = "natural_plus"
                    if getattr(profile, "is_low_light", False):
                        pn_auto = "low_light"
                    elif getattr(profile, "is_overexposed", False):
                        pn_auto = "natural"
                    elif faces_probe >= 2:
                        pn_auto = "group"
                    elif faces_probe == 1:
                        # When quality is poor, use portrait; otherwise stick to the
                        # safe default phone look.
                        if getattr(profile, "is_blurry", False) or getattr(profile, "is_noisy", False):
                            pn_auto = "portrait"
                        else:
                            pn_auto = "natural_plus"
                    preset = PRESETS.get(pn_auto) or preset
                    if preset is not None:
                        log.append("  [AUTO] preset selected: '{}'".format(preset.name))
                except Exception:
                    pass

            if preset is not None and fidelity_weight is None:
                fidelity_weight = float(getattr(preset, "fidelity", 0.5))

            plan = _timeit(
                "decide",
                lambda: self._engine.decide(
                    profile,
                    faces_found=faces_probe,
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

        # ── Plan guardrails (webcam-friendly): avoid double AI detail stacks ──
        # If GFPGAN already ran for a detected face, running Real-ESRGAN right after
        # often makes the output look crunchy / artificial on webcam captures.
        # Keep Real-ESRGAN only when it's truly needed (very blur/noise/low contrast/low light).
        if plan is not None and profile is not None:
            try:
                pn_now = str(getattr(preset, "name", "") or "").lower() if preset is not None else ""
                gf_on = any(d.layer == Layer.GFPGAN and d.run for d in plan.steps)
                if gf_on:
                    # Webcams often score as "blurry" even when the face is usable.
                    # If a face is present and GFPGAN already ran, prefer NOT stacking
                    # Real-ESRGAN unless quality is truly poor.
                    faces_present = bool(faces_probe > 0)
                    very_blur = float(getattr(profile, "blur_score", 0.0)) < 40.0
                    low_light = bool(getattr(profile, "is_low_light", False))
                    noisy = bool(getattr(profile, "is_noisy", False))
                    low_contrast = float(getattr(profile, "contrast", 1.0)) < 0.16
                    # Keep Real-ESRGAN only for extreme cases (or when no face exists).
                    keep = (not faces_present) and (very_blur or low_light or noisy or low_contrast)
                    keep = keep or (faces_present and (low_light or noisy or low_contrast or very_blur))
                    if not keep:
                        for d in plan.steps:
                            if d.layer == Layer.REAL_ESRGAN and d.run:
                                d.run = False
                                log.append("  [AUTO] REAL_ESRGAN suppressed — GFPGAN already ran (avoid over-processing / latency)")
                                break
                # When we mutate the plan, refresh the summary line (small UX polish).
                plan.steps.sort(key=lambda d: d.priority)
                run_layers = [d.layer.value for d in plan.steps if d.run]
                plan.summary = ("All layers skipped - image already clean."
                                if not run_layers
                                else "Execution plan: " + " -> ".join(run_layers))
            except Exception:
                pass

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
                        large_frame = mp_in >= 1.0
                        if large_frame and not (very_blur or low_light or noisy or low_contrast):
                            d.run = False
                            log.append("  [WOW] REAL_ESRGAN suppressed (perf) — large frame, rely on Global+Region")
                        elif not (very_blur or low_light or noisy or low_contrast):
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
                elif pn == "wow":
                    # WOW should be strong but safe: prefer blending back when drift grows.
                    if label == "GlobalEnhance":
                        guard = QualityGuard(accept_threshold=72.0, warn_threshold=60.0, blend_ratio=0.75)
                    elif label == "RegionEnhance":
                        guard = QualityGuard(accept_threshold=72.0, warn_threshold=60.0, blend_ratio=0.70)
                    elif label == "GFPGAN":
                        guard = QualityGuard(accept_threshold=70.0, warn_threshold=58.0, blend_ratio=0.60)
                    elif label == "RealESRGAN":
                        guard = QualityGuard(accept_threshold=70.0, warn_threshold=58.0, blend_ratio=0.55)

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
            # IMPORTANT (quality): on ~720p webcam captures, a 320px proxy causes visible
            # pixelation on faces. Only enable proxy cosmetics on truly large frames.
            fast_cosmetic = (pn_preset == "wow") and (mp_in >= 2.0)
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
                        reg_cfg = _scale_region_cfg(reg_cfg, k_auto)
                        reg_enh = RegionEnhancer(reg_cfg)
                        log.append("  [WOW] dynamic region cfg applied")
                    else:
                        reg_cfg = _scale_region_cfg(_region_cfg_from_preset(preset), k_auto)
                        # Backlit portraits: reduce "plastic" risk by dialing cosmetics down a bit.
                        try:
                            hi_reg = _highlight_ratio(current)
                            if hi_reg > 0.10 and pn_cfg in ("portrait", "natural_plus"):
                                reg_cfg = _scale_region_cfg(reg_cfg, 0.85)
                                # Extra targeted clamps for the most "filter-looking" knobs.
                                reg_cfg.skin_smooth = _clamp01(reg_cfg.skin_smooth * 0.90)
                                reg_cfg.eye_sharpen = _clamp01(reg_cfg.eye_sharpen * 0.85)
                                reg_cfg.lip_vibrance = _clamp01(reg_cfg.lip_vibrance * 0.85)
                                log.append("  [AUTO] backlit cosmetic clamp applied (hi={:.3f})".format(hi_reg))
                        except Exception:
                            pass
                        reg_enh = RegionEnhancer(reg_cfg)
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

            # ── Step 5b: Face-only micro polish (clean frames) ──
            # Even when the image is "clean", we want a subtle, phone-like crispness
            # on the face without affecting the background.
            try:
                pn_pol = str(getattr(preset, "name", "") or "").lower() if preset is not None else ""
                gf_ran = bool(result.restoration is not None and getattr(result.restoration, "faces_found", 0) > 0)
                if (pn_pol in ("natural_plus", "portrait")) and (not gf_ran) and (strength <= 0.15):
                    outp = _timeit("face_polish", lambda: _apply_face_micro_polish(current, result.semantic, amount=(0.22 - 0.10 * strength)))
                    if outp is not None:
                        log.append("  [AUTO] face micro-polish applied")
                        current = _apply_guard(current, outp, "FacePolish", None)
            except Exception:
                pass

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
                    glob_enh = GlobalEnhancer(_scale_global_cfg(_low_light_natural_global_cfg(profile), k_auto))
                    tag = "low-light" if getattr(profile, "is_low_light", False) else "blurry"
                    log.append("  [PRESET] mild global ({}) for '{}'".format(tag, pn))
                elif preset is not None and _preset_has_global_overrides(preset):
                    pn2 = str(getattr(preset, "name", "") or "").lower()
                    if pn2 == "wow":
                        glob_enh = GlobalEnhancer(_scale_global_cfg(_wow_dynamic_global_cfg(preset, profile), k_auto))
                        log.append("  [WOW] dynamic global cfg applied")
                    else:
                        glob_enh = GlobalEnhancer(_scale_global_cfg(_global_cfg_from_preset(preset), k_auto))
                        log.append("  [PRESET] global overrides from '{}'".format(
                            preset.name))
                else:
                    # Do not mutate shared cfg; use a scaled copy for this run.
                    glob_enh = GlobalEnhancer(_scale_global_cfg(self._global_enhancer.cfg, k_auto))

                skip_global = False
                face_only_polish = False
                # Backlit/highlight scenes: clamp local contrast / tone mapping to avoid
                # curtain/window banding and colour shifts.
                try:
                    hi = _highlight_ratio(current)
                    pn_live = str(getattr(preset, "name", "") or "").lower() if preset is not None else ""
                    faces_present = bool(getattr(result, "semantic", None) is not None and getattr(result.semantic, "faces", None))
                    # Portrait-style presets must never wreck the background. Default to
                    # face-only global polish even when not backlit.
                    if (pn_live in ("portrait", "natural_plus")) and faces_present and (profile is not None) and (not getattr(profile, "is_low_light", False)):
                        face_only_polish = True
                        log.append("  [AUTO] portrait-safe global: face-only polish")
                    # Hard skip for very backlit scenes unless we're actually in low-light.
                    # This prevents the "broken / posterized curtain" look.
                    if hi > 0.10 and profile is not None and (not getattr(profile, "is_low_light", False)):
                        log.append("  [AUTO] backlit scene (hi={:.3f}) — GLOBAL_ENHANCE skipped".format(hi))
                        skip_global = True
                        # If we have a face mask (semantic ran), we can still apply a
                        # very mild polish only on the face area without damaging
                        # the curtains/windows.
                        if result.semantic is not None and getattr(result.semantic, "faces", None):
                            face_only_polish = True
                    if (not skip_global) and hi > 0.03:
                        cfg = glob_enh.cfg
                        cfg.clahe_strength = float(np.clip(cfg.clahe_strength * 0.60, 0.0, 1.0))
                        cfg.hdr_tone       = float(np.clip(cfg.hdr_tone * 0.65, 0.0, 1.0))
                        cfg.sharpen        = float(np.clip(cfg.sharpen * 0.75, 0.0, 1.0))
                        cfg.vibrance       = float(np.clip(cfg.vibrance * 0.60, 0.0, 1.0))
                        cfg.shadow_lift    = float(np.clip(cfg.shadow_lift * 0.85, 0.0, 1.0))
                        cfg.white_balance  = float(np.clip(cfg.white_balance * 0.85, 0.0, 1.0))
                        cfg.film_look      = float(np.clip(cfg.film_look * 0.85, 0.0, 1.0))
                        log.append("  [AUTO] backlit clamp applied (hi={:.3f})".format(hi))
                except Exception:
                    pass

                if skip_global:
                    if not face_only_polish:
                        # Keep the pipeline output as-is; guard would be redundant.
                        log.append("  GlobalEnhancer: passthrough (skipped)")
                        raise StopIteration()
                    log.append("  [AUTO] face-only polish in backlit scene")

                # On huge images, run the global stack on a proxy and upscale back.
                base_global = current
                if perf_large:
                    base_global = _resize_max_side(current, 1280)
                    log.append("  [PERF] global proxy: {}x{} -> {}x{}".format(
                        current.shape[1], current.shape[0], base_global.shape[1], base_global.shape[0]
                    ))

                ge = _timeit(
                    "global",
                    lambda: glob_enh.enhance(base_global, profile=profile, use_temporal=False),
                )
                if ge.success and ge.image is not None:
                    log.extend("  " + ln for ln in ge.log)
                    out_img = ge.image
                    if perf_large and out_img.shape[:2] != current.shape[:2]:
                        try:
                            out_img = cv2.resize(out_img, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_LINEAR)
                        except Exception:
                            out_img = ge.image
                    if face_only_polish and result.semantic is not None:
                        # Blend only into the face region (soft feather) to avoid background artifacts.
                        try:
                            # Extra safety: in strong backlit, keep the face-only polish very mild
                            # (avoid "makeup filter" feel).
                            try:
                                hi_face = _highlight_ratio(current)
                                if hi_face > 0.10:
                                    cfg = glob_enh.cfg
                                    cfg.clahe_strength = _clamp01(cfg.clahe_strength * 0.65)
                                    cfg.hdr_tone       = _clamp01(cfg.hdr_tone * 0.70)
                                    cfg.sharpen        = _clamp01(cfg.sharpen * 0.75)
                                    cfg.vibrance       = _clamp01(cfg.vibrance * 0.70)
                                    cfg.film_look      = _clamp01(cfg.film_look * 0.55)
                                    log.append("  [AUTO] backlit face-polish soften (hi={:.3f})".format(hi_face))
                            except Exception:
                                pass
                            mask = None
                            face0 = result.semantic.faces[0]
                            skin = None
                            try:
                                skin = face0.regions.get("skin")
                            except Exception:
                                skin = None
                            if skin is not None and getattr(skin, "mask", None) is not None:
                                mask = (skin.mask.astype(np.float32) / 255.0)
                            elif getattr(result.semantic, "label_map", None) is not None:
                                mask = (result.semantic.label_map > 0).astype(np.float32)
                            if mask is not None:
                                if mask.shape[:2] != current.shape[:2]:
                                    mask = cv2.resize(mask, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_LINEAR)
                                mask = cv2.GaussianBlur(mask, (0, 0), 9.0)
                                mask = np.clip(mask, 0.0, 1.0)
                                blended = (
                                    current.astype(np.float32) * (1.0 - mask[..., None]) +
                                    out_img.astype(np.float32) * mask[..., None]
                                ).astype(np.uint8)
                                current = _apply_guard(current, blended, "GlobalEnhance(face)", None)
                            else:
                                current = _apply_guard(current, out_img, "GlobalEnhance", None)
                        except Exception:
                            current = _apply_guard(current, out_img, "GlobalEnhance", None)
                    else:
                        current = _apply_guard(current, out_img, "GlobalEnhance", None)
                else:
                    log.append("  GlobalEnhancer returned no result -- passthrough")
            except StopIteration:
                pass
            except Exception as _e:
                log.append("  [WARN] GlobalEnhancer failed: {}".format(_e))

        result.restored = current

        # Step 6: Quality Metrics
        log.append("--- Quality Metrics ---")
        try:
            # Metrics are for diagnostics; compute on proxy for huge images.
            if perf_large:
                img_a = _resize_max_side(image, 1024)
                img_b = _resize_max_side(result.restored, 1024)
                log.append("  [PERF] metrics proxy: {}x{} -> {}x{}".format(
                    image.shape[1], image.shape[0], img_a.shape[1], img_a.shape[0]
                ))
                m = _timeit("metrics", lambda: self._metrics.compute_all(img_a, img_b))
            else:
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
            log.append("  ΔE (LAB)     = {:.2f}".format(getattr(m, "delta_e", 0.0)))
            log.append("  ΔS (HSV)     = {:.1f}".format(getattr(m, "sat_shift", 0.0)))
        except Exception as e:
            log.append("[WARN] Metrics failed: {}".format(e))

        # Step 7: Difference map (always — large inputs use a proxy then upscale heat)
        try:
            if perf_large:
                da = _resize_max_side(image, 1024)
                db = _resize_max_side(result.restored, 1024)
                dm_small = _timeit(
                    "diff_map", lambda: self._metrics.difference_map(da, db))
                h0, w0 = image.shape[:2]
                result.diff_map = cv2.resize(
                    dm_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
                log.append(
                    "  Difference map: proxy {}x{} → full {}x{} (heatmap)".format(
                        dm_small.shape[1], dm_small.shape[0], w0, h0))
            else:
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
