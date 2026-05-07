"""
Decision Engine — chooses *which* enhancement layers to run for an image.

CMPE 491 Senior Design Project.

This is the explicit, deterministic, audit-ready logic the High-Level
Design report calls for ("explainable processing, where every enhancement
step is deterministic, traceable, and justifiable" — HLD §1).

The pipeline previously executed every layer unconditionally; that
contradicts the doc's central thesis ("AI as a guided tool, not the
boss") and burns CPU on no-op work. The Decision Engine reads a
ProfileResult plus optional face-detection metadata and emits a typed
plan that the pipeline executes step by step, logging the *reason* for
each decision.

Public API
----------
    plan = DecisionEngine().decide(profile, faces_found=N, image_shape=(H,W,C))
    plan.steps            # list[Decision] in execution order
    plan.justification    # human-readable explanation block
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from profiler import ProfileResult


# ── Layer enum ────────────────────────────────────────────────────────

class Layer(str, Enum):
    """Pipeline layers known to the decision engine."""
    GFPGAN          = "GFPGAN"           # face_restorer.py
    REAL_ESRGAN     = "REAL_ESRGAN"      # general_restorer.py
    FACE_PARSE      = "FACE_PARSE"       # semantic_parser.py
    REGION_ENHANCE  = "REGION_ENHANCE"   # region_enhancer.py
    GLOBAL_ENHANCE  = "GLOBAL_ENHANCE"   # global_enhancer.py


# ── Decision records ──────────────────────────────────────────────────

@dataclass
class Decision:
    """One layer decision with its supporting reason."""
    layer:    Layer
    run:      bool
    reason:   str
    priority: int = 0              # for ordering within the plan


@dataclass
class Plan:
    """Ordered list of layer decisions for one image."""
    steps:   List[Decision] = field(default_factory=list)
    summary: str = ""

    @property
    def to_run(self) -> List[Layer]:
        return [d.layer for d in self.steps if d.run]

    def justification(self) -> List[str]:
        """Human-readable lines for the pipeline log."""
        out: List[str] = []
        for d in self.steps:
            mark = "RUN " if d.run else "SKIP"
            out.append("  [{}] {:<14} -- {}".format(mark, d.layer.value, d.reason))
        if self.summary:
            out.append("  >> {}".format(self.summary))
        return out


# ── Engine ────────────────────────────────────────────────────────────

@dataclass
class EngineThresholds:
    """All numeric thresholds gathered in one place for auditability."""
    blur_sharp:       float = 250.0   # >= this is considered already sharp
    blur_acceptable:  float = 60.0    # below this we definitely run AI
    noise_clean:      float = 4.0     # below this skip denoising
    noise_high:       float = 12.0    # above this trigger Real-ESRGAN
    skin_face_min:    float = 0.04    # below this assume no face
    low_res_mp:       float = 1.5     # below this prefer Real-ESRGAN x2
    # Real-ESRGAN now downsamples before AI when input is huge, so we
    # don't need a hard upper bound any more. 50 MP is a safety net for
    # absurd cases (RAW photos, scans, etc) where even downsampling is
    # not worth the wait.
    high_res_mp:      float = 50.0


class DecisionEngine:
    """
    Profile-driven layer selector.

    The engine never executes anything itself; it just produces a plan
    that the pipeline runs. This keeps the decision logic standalone and
    fully unit-testable.
    """

    def __init__(self, thr: Optional[EngineThresholds] = None) -> None:
        self.thr = thr or EngineThresholds()

    # ── public API ───────────────────────────────────────────────────

    def decide(
        self,
        profile: Optional[ProfileResult],
        faces_found: int = 0,
        image_shape: Optional[Tuple[int, int, int]] = None,
        preset_name: Optional[str] = None,
    ) -> Plan:
        """Build an execution Plan."""
        plan = Plan()

        if profile is None:
            plan.steps = [
                Decision(Layer.GLOBAL_ENHANCE, True,
                         "no profile available — apply safe global pass only", 50),
            ]
            plan.summary = "Fallback: global only."
            return plan

        h, w = (image_shape[:2] if image_shape else (0, 0))
        mp = (h * w) / 1e6 if (h and w) else 0.0
        thr = self.thr
        pn = (preset_name or "").strip().lower()
        # Preset-aware threshold tuning: keep it simple and audit-friendly.
        # WOW/MAGAZINE/VIDID aim for stronger visible effects, so we:
        # - treat "already sharp" more strictly (lower blur_sharp)
        # - allow cosmetics more often (even if looks_fine)
        if pn in ("wow", "magazine", "vivid"):
            thr = EngineThresholds(
                blur_sharp=180.0,
                blur_acceptable=55.0,
                noise_clean=thr.noise_clean,
                noise_high=thr.noise_high,
                skin_face_min=thr.skin_face_min,
                low_res_mp=thr.low_res_mp,
                high_res_mp=thr.high_res_mp,
            )

        has_face = (faces_found > 0) or (
            profile.has_skin and profile.skin_ratio >= thr.skin_face_min)
        is_blurry = profile.blur_score < thr.blur_sharp
        very_blurry = profile.blur_score < thr.blur_acceptable
        is_noisy = profile.noise_level > thr.noise_clean
        very_noisy = profile.noise_level > thr.noise_high
        low_res = (mp > 0) and (mp < thr.low_res_mp)
        too_big_for_sr = (mp > 0) and (mp > thr.high_res_mp)

        # ── Layer 1: GFPGAN — face restoration ──
        if has_face and is_blurry:
            plan.steps.append(Decision(
                Layer.GFPGAN, True,
                "face detected (skin={:.2f}) AND image is not perfectly sharp "
                "(blur={:.0f} < {:.0f})".format(
                    profile.skin_ratio, profile.blur_score, thr.blur_sharp),
                priority=10))
        elif has_face and not is_blurry:
            plan.steps.append(Decision(
                Layer.GFPGAN, False,
                "face present but already sharp (blur={:.0f} >= {:.0f}) — "
                "no AI restoration needed".format(
                    profile.blur_score, thr.blur_sharp),
                priority=10))
        else:
            plan.steps.append(Decision(
                Layer.GFPGAN, False,
                "no face detected (skin_ratio={:.3f} < {:.2f})".format(
                    profile.skin_ratio, thr.skin_face_min),
                priority=10))

        # ── Layer 2: Real-ESRGAN — general AI cleanup ──
        # GeneralRestorer does AI denoising + sharpening at the
        # original size (any upscale is cancelled internally and the
        # input is auto-downsampled if huge), so it's cheap enough to
        # run on almost everything that isn't already pristine.
        run_general = False
        general_reason = ""
        if too_big_for_sr:
            general_reason = (
                "image absurdly large ({:.1f} MP > {:.1f} MP) -- skip "
                "to avoid extreme latency".format(mp, thr.high_res_mp))
        elif (
            profile.is_blurry
            or is_noisy
            or profile.is_low_light
            or (low_res and (very_blurry or very_noisy or profile.contrast < 0.18))
        ):
            run_general = True
            general_reason = (
                "AI cleanup useful (blurry={}, noisy={}, low_res={})".format(
                    profile.is_blurry, is_noisy, low_res))
        elif profile.contrast < 0.18:
            run_general = True
            general_reason = (
                "low contrast ({:.2f}) -- Real-ESRGAN micro-detail recovery "
                "helps".format(profile.contrast))
        else:
            general_reason = (
                "image already crisp + clean (blur={:.0f}, noise={:.1f}, "
                "contrast={:.2f}) -- AI cleanup unnecessary".format(
                    profile.blur_score, profile.noise_level, profile.contrast))
        plan.steps.append(Decision(
            Layer.REAL_ESRGAN, run_general, general_reason, priority=20))

        # ── Layer 3: Face Parse + Region Enhance (cosmetic per-region) ──
        #
        # This step is what gives the "phone beauty" look, but it can also
        # make already-good modern photos look over-processed. Therefore:
        # - If the image already looks fine (not low-light, not noisy, not blurry),
        #   we SKIP cosmetics by default. Users can still force it via presets.
        looks_fine = (
            not profile.is_low_light and
            not profile.is_overexposed and
            not profile.is_blurry and
            not profile.is_noisy
        )
        if pn in ("wow", "magazine", "vivid"):
            run_cosmetic = has_face and (not profile.is_overexposed)
        else:
            run_cosmetic = has_face and (not looks_fine)

        if has_face:
            plan.steps.append(Decision(
                Layer.FACE_PARSE, run_cosmetic,
                "face present → parse needed for cosmetics" if run_cosmetic else
                "face present but image already clean — skip cosmetic parsing by default",
                priority=30))
            plan.steps.append(Decision(
                Layer.REGION_ENHANCE, run_cosmetic,
                "apply per-region cosmetic enhancement" if run_cosmetic else
                "cosmetics skipped — preserve natural rendering",
                priority=40))
        else:
            plan.steps.append(Decision(
                Layer.FACE_PARSE, False,
                "no face → semantic parse would produce empty masks",
                priority=30))
            plan.steps.append(Decision(
                Layer.REGION_ENHANCE, False,
                "depends on FACE_PARSE — cascade skip",
                priority=40))

        # ── Layer 4: Global Enhancement — only when image actually needs it ──
        # GlobalEnhancer applies an aggressive multi-stage stack (white
        # balance + shadow lift + CLAHE + HDR + vibrance + film grade).
        # Running it on a clean, well-lit, in-focus image overcooks the
        # output (purple/lavender wash, exaggerated saturation). Skip it
        # in those cases to preserve fidelity.
        if profile.is_overexposed:
            run_global = False
            global_reason = (
                "image already overexposed (brightness={:.2f}) -- global stack "
                "would push tones further; skip to keep fidelity".format(
                    profile.brightness))
        elif looks_fine and pn not in ("wow", "magazine", "vivid"):
            run_global = False
            global_reason = (
                "image already well-balanced (bright/contrast/noise/blur all OK) "
                "-- skipping global pass to avoid over-processing")
        else:
            run_global = True
            global_reason = (
                "image needs global help "
                "(low_light={}, blurry={}, noisy={})".format(
                    profile.is_low_light, profile.is_blurry, profile.is_noisy))
        plan.steps.append(Decision(
            Layer.GLOBAL_ENHANCE, run_global, global_reason, priority=50))

        # Stable order by priority
        plan.steps.sort(key=lambda d: d.priority)

        # Summary line
        run_layers = [d.layer.value for d in plan.steps if d.run]
        if not run_layers:
            plan.summary = "All layers skipped - image already clean."
        else:
            plan.summary = "Execution plan: " + " -> ".join(run_layers)
        return plan
