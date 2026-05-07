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
from typing import List, Optional

import numpy as np

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

        if image is None or image.size == 0:
            log.append("[ERROR] Invalid input")
            return result

        log.append("=== ImageEnhancementPipeline ===")
        log.append("Input: {}x{} ({:.1f} MP)".format(
            image.shape[1], image.shape[0],
            image.shape[0] * image.shape[1] / 1e6))

        # ── Step 1: analyzeImage() ──
        profile = self.analyzeImage(image)
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
            plan = self._engine.decide(profile, faces_found=0,
                                       image_shape=image.shape)
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
                if d.layer == Layer.REAL_ESRGAN:
                    if preset.skip_general and d.run:
                        d.run = False
                        log.append("  [PRESET] REAL_ESRGAN forced OFF "
                                   "by '{}' preset".format(preset.name))
                    elif preset.force_general and not d.run:
                        d.run = True
                        log.append("  [PRESET] REAL_ESRGAN forced ON "
                                   "by '{}' preset".format(preset.name))
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
            rep = self._guard.evaluate(before, after, label=label)
            result.guard_reports.append(rep)
            log.append(rep.log[0])
            return rep.output

        # ── Step 3: GFPGAN — face restoration ──
        if _layer_should_run(Layer.GFPGAN):
            log.append("--- GFPGAN Restoration ---")
            resto = self._restorer.restore(
                current,
                fidelity_weight=fidelity_weight,
                only_center_face=only_center_face,
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
            gr = self._general.restore(current)
            result.general_result = gr
            log.extend(gr.log)
            if gr.success and gr.restored is not None:
                current = _apply_guard(
                    current, gr.restored, "RealESRGAN", "general_result")

        # ── Step 5: Face parse + region cosmetic ──
        if _layer_should_run(Layer.FACE_PARSE):
            log.append("--- Semantic Layer ---")
            sem = self._parser.parse(current)
            result.semantic = sem
            log.extend("  " + ln for ln in sem.log)

            if sem.success and sem.faces and _layer_should_run(Layer.REGION_ENHANCE):
                # Preset overrides the per-region knobs; without one
                # we use the pipeline's default RegionEnhancer so user
                # preferences from __init__ are preserved.
                if preset is not None:
                    reg_enh = RegionEnhancer(_region_cfg_from_preset(preset))
                    log.append("  [PRESET] region knobs from '{}' "
                               "(skin={:.2f} eyes={:.2f} lips={:.2f})".format(
                                   preset.name, preset.skin_amount,
                                   preset.eyes_sharpen, preset.lips_vibrance))
                else:
                    reg_enh = self._enhancer
                reg = reg_enh.apply(current, sem)
                result.region_result = reg
                log.extend("  " + ln for ln in reg.log)
                if reg.success and reg.image is not None:
                    current = _apply_guard(
                        current, reg.image, "RegionEnhance", "region_result")

        # ── Step 6: Global Enhancement (Furkan) ──
        if _layer_should_run(Layer.GLOBAL_ENHANCE):
            log.append("--- GlobalEnhancer ---")
            try:
                # Preset overrides the global knobs (WB, vibrance,
                # sharpen, …). When absent, use the shared instance so
                # the temporal frame buffer stays consistent.
                if preset is not None:
                    glob_enh = GlobalEnhancer(_global_cfg_from_preset(preset))
                    log.append("  [PRESET] global knobs from '{}'".format(
                        preset.name))
                else:
                    glob_enh = self._global_enhancer
                ge = glob_enh.enhance(
                    current, profile=profile, use_temporal=False)
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
            m = self._metrics.compute_all(image, result.restored)
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
            result.diff_map = self._metrics.difference_map(image, result.restored)
            log.append("  Difference map: generated")
        except Exception as e:
            log.append("[WARN] Diff map failed: {}".format(e))

        result.success = True
        log.append("=== Pipeline completed ===")
        return result
