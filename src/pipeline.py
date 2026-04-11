"""
ImageEnhancementPipeline — GFPGAN-based System.

CMPE 491 Senior Design Project.

Processing sequence (GFPGAN-centric):
    1. analyzeImage()   — ImageProfiler: measure scene characteristics
    2. detectFaces()    — FaceRestorer: locate faces in image
    3. restoreFaces()   — FaceRestorer: GFPGAN v1.3 AI restoration
    4. [GlobalEnhancer] — Team member's module (exposure, color, contrast)
    5. computeMetrics() — QualityMetrics: PSNR, SSIM, etc.
    6. diffMap()        — Visual explanation of changes

Team responsibilities:
    Pixel Layer (this module):  face detection + GFPGAN restoration
    Global Layer (team member): exposure, color balance, contrast
    Semantic Layer (team):      segmentation, labeling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from profiler import ImageProfiler, ProfileResult
from face_restorer import FaceRestorer, RestorationResult
from metrics import MetricsCalculator, QualityMetrics


@dataclass
class EnhancementResult:
    """Complete output from one pipeline run."""
    original:       Optional[np.ndarray] = None
    restored:       Optional[np.ndarray] = None
    diff_map:       Optional[np.ndarray] = None
    profile:        Optional[ProfileResult] = None
    metrics:        Optional[QualityMetrics] = None
    restoration:    Optional[RestorationResult] = None
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

    # ── public API ────────────────────────────────────────────────

    def analyzeImage(self, image: np.ndarray) -> Optional[ProfileResult]:
        """Evaluate image: lighting, noise, blur, skin presence."""
        return self._profiler.profile(image)

    def restoreImage(
        self,
        image: np.ndarray,
        fidelity_weight: Optional[float] = None,
        only_center_face: bool = False,
    ) -> EnhancementResult:
        """
        Full pipeline: profile → GFPGAN restore → metrics → diff map.

        Parameters
        ----------
        image : np.ndarray      BGR uint8 input
        fidelity_weight : float 0=max AI, 1=original (override instance default)
        only_center_face : bool Process only the most central face
        """
        result = EnhancementResult(original=image.copy() if image is not None else None)
        log = result.log

        if image is None or image.size == 0:
            log.append("[ERROR] Invalid input")
            return result

        log.append("=== ImageEnhancementPipeline ===")
        log.append("Input: {}x{} ({:.1f} MP)".format(
            image.shape[1], image.shape[0],
            image.shape[0] * image.shape[1] / 1e6))

        # Step 1: analyzeImage()
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
            log.append("  scene flags  : {}".format(profile.summary()))

        # Step 2+3: detectFaces() + restoreFaces() via FaceRestorer
        log.append("--- GFPGAN Restoration ---")
        resto = self._restorer.restore(
            image,
            fidelity_weight=fidelity_weight,
            only_center_face=only_center_face,
        )
        result.restoration = resto
        log.extend(resto.log)
        log.append("  Faces found : {}".format(resto.faces_found))

        if not resto.success or resto.restored is None:
            log.append("[ERROR] Restoration failed — returning original")
            result.restored = image
            result.success = False
            return result

        result.restored = resto.restored

        # Step 4: [GlobalEnhancer placeholder]
        log.append("--- GlobalEnhancer ---")
        log.append("  [NOT ACTIVE] Team member's module (exposure, color, contrast)")

        # Step 5: Quality Metrics
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

        # Step 6: Difference map
        try:
            result.diff_map = self._metrics.difference_map(image, result.restored)
            log.append("  Difference map: generated")
        except Exception as e:
            log.append("[WARN] Diff map failed: {}".format(e))

        result.success = True
        log.append("=== Pipeline completed ===")
        return result
