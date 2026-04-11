"""
ImageProcessor — Central Orchestrator for the Enhancement Pipeline.

Analysis Report (Section 3.5.3, Class #3):
    ImageProcessor
        - analyzeImage()   : Evaluates lighting, noise, blur, and patterns
        - enhanceImage()   : Coordinates all enhancement modules

    Relationships:
        - Aggregates: Image, LabelBasedProcessor, PixelBasedProcessor, GeneralEnhancer

Analysis Report (Section 3.5.4, Dynamic Model / Sequence):
    1. User → System: Upload image
    2. System → ImageProcessor: Pass raw image
    3. ImageProcessor → analyzeImage(): Determine enhancement needs
    4. ImageProcessor → LabelBasedProcessor: detectObjects() [future]
    5. ImageProcessor → PixelBasedProcessor: reduceNoise(), sharpenImage()
    6. ImageProcessor → GeneralEnhancer: [future, friend's module]
    7. ImageProcessor → System: Return enhanced image

    Alternative Behaviors (HLD 3.2.5, Analysis Report):
        - If pixel enhancement fails → fallback to original
        - If any processor returns error → notify and use fallback

NOTE: Currently only Pixel module is active (our responsibility).
      LabelBasedProcessor and GeneralEnhancer will be integrated
      when team members deliver their modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from profiler import ImageProfiler, ProfileResult
from pixel_enhance import PixelBasedProcessor, ProcessingLog
from metrics import MetricsCalculator, QualityMetrics


@dataclass
class EnhancementResult:
    """Complete result from ImageProcessor.enhanceImage().

    Contains all artifacts needed for display, comparison, and reporting.
    """

    original: Optional[np.ndarray] = None
    enhanced: Optional[np.ndarray] = None
    profile: Optional[ProfileResult] = None
    mask: Optional[np.ndarray] = None
    metrics: Optional[QualityMetrics] = None
    difference_map: Optional[np.ndarray] = None
    log: List[str] = field(default_factory=list)
    success: bool = False


def _make_soft_center_mask(h: int, w: int) -> np.ndarray:
    """Generate a soft elliptical ROI mask centered on the image."""
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    rw, rh = int(w * 0.25), int(h * 0.25)
    cv2.ellipse(mask, (cx, cy), (rw, rh), 0, 0, 360, 1.0, thickness=-1)

    kx = max(3, (w // 20) | 1)
    ky = max(3, (h // 20) | 1)
    mask = cv2.GaussianBlur(mask, (kx, ky), 0)

    m_min, m_max = float(mask.min()), float(mask.max())
    if m_max > m_min:
        mask = (mask - m_min) / (m_max - m_min)
    return mask.astype(np.float32)


class ImageProcessor:
    """Central controller class that orchestrates the enhancement pipeline.

    Analysis Report (Section 3.5.3, Class #3):
        - analyzeImage() → delegates to ImageProfiler
        - enhanceImage() → coordinates PixelBasedProcessor
                           (and future LabelBasedProcessor, GeneralEnhancer)

    HLD (Section 3.2.5 - Orchestration and Control):
        - Sequencing: Semantic → Pixel → Global
        - Error Handling: fallback if a module fails
        - Real-time Decisions: profile-based module activation
    """

    def __init__(
        self,
        roi_blend: float = 0.7,
        bg_blend: float = 0.5,
        clahe_clip: float = 2.0,
    ) -> None:
        self._profiler = ImageProfiler()
        self._pixel_processor = PixelBasedProcessor(
            roi_blend=roi_blend,
            bg_blend=bg_blend,
            clahe_clip=clahe_clip,
        )
        self._metrics = MetricsCalculator()

    # ------------------------------------------------------------------
    #  analyzeImage()  — Analysis Report 3.5.3
    #  "Evaluates lighting, noise, blur, and detectable patterns"
    # ------------------------------------------------------------------

    def analyzeImage(self, image: np.ndarray) -> Optional[ProfileResult]:
        """Analyze input image to determine processing parameters.

        Delegates to ImageProfiler which measures:
        brightness, contrast, blur, noise, edge density, skin presence.
        """
        return self._profiler.profile(image)

    # ------------------------------------------------------------------
    #  enhanceImage()  — Analysis Report 3.5.3
    #  "Coordinates all enhancement modules"
    #
    #  Sequence (Analysis Report 3.5.4):
    #    analyzeImage() → [LabelBased] → PixelBased → [GeneralEnhancer]
    # ------------------------------------------------------------------

    def enhanceImage(
        self,
        image: np.ndarray,
        mask_map: Optional[np.ndarray] = None,
        *,
        auto_mask: bool = True,
        manual_decisions: Optional[Dict] = None,
    ) -> EnhancementResult:
        """Run the full enhancement pipeline on a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR, uint8, (H,W,3) input.
        mask_map : np.ndarray or None
            Mask from LabelBasedProcessor. If None and auto_mask=True,
            a soft center-ROI is generated for testing.
        auto_mask : bool
            Generate auto mask when mask_map is None.
        manual_decisions : dict or None
            If provided, bypasses auto decision engine and uses user
            slider values. Enables GUI manual mode.

        Returns
        -------
        EnhancementResult with all outputs and explainable log.
        """
        result = EnhancementResult()
        log = result.log

        # --- Input Validation ---
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            log.append("[ERROR] Invalid input image")
            return result

        if image.ndim != 3 or image.shape[2] != 3:
            log.append("[ERROR] Expected BGR (H,W,3) image")
            return result

        result.original = image.copy()
        h, w = image.shape[:2]
        log.append("Input: {}x{} ({:.1f} MP)".format(w, h, w * h / 1e6))

        # --- Step 1: analyzeImage() ---
        profile = self.analyzeImage(image)
        if profile is None:
            log.append("[ERROR] analyzeImage() failed — returning original")
            result.enhanced = image
            return result

        result.profile = profile
        log.append("analyzeImage():")
        log.append("  brightness    = {:.3f} {}".format(
            profile.brightness,
            "(LOW)" if profile.is_low_light else
            "(HIGH)" if profile.is_overexposed else "(OK)"
        ))
        log.append("  contrast      = {:.3f}".format(profile.contrast))
        log.append("  blur_score    = {:.1f} {}".format(
            profile.blur_score,
            "(BLURRY)" if profile.is_blurry else "(SHARP)"
        ))
        log.append("  noise_level   = {:.1f} {}".format(
            profile.noise_level,
            "(NOISY)" if profile.is_noisy else "(CLEAN)"
        ))
        log.append("  edge_density  = {:.4f}".format(profile.edge_density))
        log.append("  skin_ratio    = {:.3f} {}".format(
            profile.skin_ratio,
            "(SKIN DETECTED)" if profile.has_skin else ""
        ))

        # --- Step 2: LabelBasedProcessor (future — team member) ---
        if mask_map is not None:
            mask = mask_map
            log.append("LabelBasedProcessor: mask provided (external)")
        elif auto_mask:
            mask = _make_soft_center_mask(h, w)
            log.append("LabelBasedProcessor: [PLACEHOLDER] auto center-ROI mask")
        else:
            mask = None
            log.append("LabelBasedProcessor: [PLACEHOLDER] no mask (full ROI)")

        result.mask = mask

        # --- Step 3: PixelBasedProcessor ---
        log.append("--- PixelBasedProcessor ---")

        profile_dict = {
            "brightness": profile.brightness,
            "contrast": profile.contrast,
            "blur": profile.blur_score,
            "blur_score": profile.blur_score,
            "edge_density": profile.edge_density,
            "noise_level": profile.noise_level,
            "skin_ratio": profile.skin_ratio,
            "is_noisy": profile.is_noisy,
            "is_blurry": profile.is_blurry,
            "has_skin": profile.has_skin,
        }

        try:
            enhanced, pixel_log = self._pixel_processor.process(
                image, profile_dict, mask_map=mask,
                manual_decisions=manual_decisions,
            )
            log.extend(pixel_log.get_log())
            result.enhanced = enhanced
        except Exception as e:
            log.append("[ERROR] PixelBasedProcessor failed: {}".format(e))
            log.append("[FALLBACK] Returning original image (HLD 3.2.5)")
            result.enhanced = image
            result.success = False
            return result

        # --- Step 4: GeneralEnhancer (future — friend's module) ---
        log.append("--- GeneralEnhancer ---")
        log.append("GeneralEnhancer: [NOT ACTIVE] (team member's module)")

        # --- Quality Metrics ---
        log.append("--- Quality Metrics ---")
        metrics = self._metrics.compute_all(image, enhanced)
        result.metrics = metrics

        log.append("  PSNR         = {:.2f} dB".format(metrics.psnr))
        log.append("  SSIM         = {:.4f}".format(metrics.ssim))
        log.append("  Entropy      = {:.3f} -> {:.3f} (delta={:+.3f})".format(
            metrics.entropy_original, metrics.entropy_enhanced,
            metrics.entropy_enhanced - metrics.entropy_original
        ))
        log.append("  Colorfulness = {:.2f} -> {:.2f} (delta={:+.2f})".format(
            metrics.colorfulness_original, metrics.colorfulness_enhanced,
            metrics.colorfulness_enhanced - metrics.colorfulness_original
        ))

        # --- Difference Map ---
        diff_map = self._metrics.compute_difference_map(image, enhanced)
        result.difference_map = diff_map
        log.append("Difference heatmap generated.")

        result.success = True
        log.append("ImageProcessor.enhanceImage() completed successfully.")
        return result
