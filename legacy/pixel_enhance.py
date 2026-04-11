"""
Pixel-Level Processing Subsystem — PixelBasedProcessor

HYBRID approach (HLD 3.1, Poster):
    "AI is only one guided element among several enhancement layers."
    If a trained EnhanceNet model is available → AI-based enhancement.
    Otherwise → deterministic classical fallback.

Analysis Report (Section 3.5.3, Class #5):
    PixelBasedProcessor
        - reduceNoise()    : Removes noise and artifacts
        - sharpenImage()   : Restores edges and details

High Level Design Report (Section 3.2.3):
    1. Adaptive Noise Suppression
    2. Controlled Sharpening
    3. Edge-Preserving Filtering

Project Specifications Report (Section 1.1):
    "Applies noise reduction, deblurring, sharpening, and fine-grained
     detail restoration directly at the pixel matrix level"

Global brightness / color adjustments are NOT performed here;
those belong to the GeneralEnhancer / Global Layer (HLD 3.2.4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    pass


CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "enhance_net_latest.pt"
)


@dataclass
class ProcessingLog:
    """Explainable processing log — every step is traceable (HLD goal #1)."""

    entries: List[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.entries.append(msg)

    def get_log(self) -> List[str]:
        return self.entries.copy()


# ---------------------------------------------------------------------------
#  Mask utilities (for region-adaptive processing per HLD 3.2.2)
# ---------------------------------------------------------------------------

def _prepare_mask(mask_map: Optional[np.ndarray], h: int, w: int) -> np.ndarray:
    """Normalize Semantic Layer mask to single-channel [0,1] float32."""
    if mask_map is None:
        return np.ones((h, w), dtype=np.float32)

    m = np.asarray(mask_map, dtype=np.float32)

    if m.ndim == 3 and m.shape[2] == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    m = np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)

    if m.max() > 1.5:
        m /= 255.0

    return np.clip(m, 0.0, 1.0)


def _to_3ch(mask: np.ndarray) -> np.ndarray:
    """Expand (H,W) mask to (H,W,3) for element-wise BGR operations."""
    if mask.ndim == 2:
        return np.repeat(mask[:, :, None], 3, axis=2)
    return mask


# ---------------------------------------------------------------------------
#  AI Model Loader
# ---------------------------------------------------------------------------

def _load_enhance_net(checkpoint_path: str, log: ProcessingLog):
    """Try to load trained EnhanceNet model.

    Returns (model, device) or (None, None) on failure.
    """
    if not _TORCH_AVAILABLE:
        log.add("AI Model: PyTorch not installed — classical mode only")
        return None, None

    if not os.path.isfile(checkpoint_path):
        log.add("AI Model: No checkpoint found at '{}' — classical mode".format(
            os.path.basename(checkpoint_path)
        ))
        return None, None

    try:
        from models import EnhanceNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = EnhanceNet(3, 3).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        epoch = checkpoint.get("epoch", "?")
        log.add("AI Model: LOADED (epoch={}, device={})".format(epoch, device))
        return model, device
    except Exception as e:
        log.add("AI Model: Load failed ({}) — classical fallback".format(e))
        return None, None


def _ai_enhance(
    img: np.ndarray,
    model,
    device,
    mask: np.ndarray,
    roi_blend: float,
    log: ProcessingLog,
) -> np.ndarray:
    """Run EnhanceNet inference on the image, blended by ROI mask."""
    h, w = img.shape[:2]

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if pad_h > 0 or pad_w > 0:
        img_rgb = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out_tensor = model(tensor)

    out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = (np.clip(out_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    if pad_h > 0 or pad_w > 0:
        out_bgr = out_bgr[:h, :w]

    mask3 = _to_3ch(mask)
    blend = mask3 * roi_blend
    result = img.astype(np.float32) * (1.0 - blend) + out_bgr.astype(np.float32) * blend
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)

    log.add("AI Enhance: APPLIED (EnhanceNet inference, ROI-blended, roi_blend={:.2f})".format(
        roi_blend
    ))
    return result


# ---------------------------------------------------------------------------
#  PixelBasedProcessor  (Analysis Report, Class #5)
# ---------------------------------------------------------------------------

class PixelBasedProcessor:
    """Improves the overall pixel-level quality of the image.

    HYBRID approach:
        1. Analyze input → determine what the image needs
        2. If trained AI model exists → use it (guided AI, not black-box)
        3. Classical pipeline always runs:
           reduceNoise() → sharpenImage() → edgePreservingFilter()
        4. Every decision is logged with reasoning (explainable)

    Methods match the documented class model (Analysis Report 3.5.3):
        - reduceNoise()           → Adaptive Noise Suppression
        - sharpenImage()          → Controlled Sharpening + Detail Restoration
        - edgePreservingFilter()  → Edge-Preserving Filtering
        - process()               → Full pipeline
    """

    def __init__(
        self,
        roi_blend: float = 0.7,
        bg_blend: float = 0.5,
        clahe_clip: float = 2.0,
        checkpoint_path: str = CHECKPOINT_PATH,
    ) -> None:
        self.roi_blend = roi_blend
        self.bg_blend = bg_blend
        self.clahe_clip = clahe_clip
        self.checkpoint_path = checkpoint_path
        self._log = ProcessingLog()

    def get_log(self) -> List[str]:
        return self._log.get_log()

    # ------------------------------------------------------------------
    #  Input Analysis → Decision Engine (HLD 3.2.5: "Real-time Decisions")
    # ------------------------------------------------------------------

    def _analyze_and_decide(self, profile: Dict) -> Dict:
        """Analyze profile and decide processing parameters.

        This is the 'brain' that makes the system input-aware.
        Every decision is explained in the log.
        """
        decisions = {}

        brightness = float(profile.get("brightness", 0.5))
        contrast = float(profile.get("contrast", 0.2))
        blur_score = float(profile.get("blur", profile.get("blur_score", 100.0)))
        noise_level = float(profile.get("noise_level", 0.0))
        edge_density = float(profile.get("edge_density", 0.0))
        is_noisy = profile.get("is_noisy", False)
        has_skin = profile.get("has_skin", False)

        self._log.add("--- Input Analysis & Decision Engine ---")

        # Noise decision
        if is_noisy or noise_level >= 5.0:
            denoise_strength = float(np.clip(noise_level * 0.5, 3.0, 20.0))
            decisions["denoise"] = True
            decisions["denoise_h"] = denoise_strength
            if has_skin:
                decisions["denoise_roi_weight"] = 0.3
                self._log.add(
                    "  [NOISE] noise_level={:.1f} => DENOISE ON (h={:.1f}), "
                    "skin detected => gentle ROI denoising (0.3) to preserve skin texture".format(
                        noise_level, denoise_strength
                    ))
            else:
                decisions["denoise_roi_weight"] = 0.5
                self._log.add(
                    "  [NOISE] noise_level={:.1f} => DENOISE ON (h={:.1f}), "
                    "no skin => moderate ROI denoising (0.5)".format(
                        noise_level, denoise_strength
                    ))
        else:
            decisions["denoise"] = False
            self._log.add(
                "  [NOISE] noise_level={:.1f} => CLEAN, denoise SKIPPED".format(noise_level)
            )

        # Sharpening decision
        if blur_score < 50.0:
            decisions["sharpen_strength"] = 1.5 + (50.0 - blur_score) / 40.0
            decisions["sharpen_sigma"] = 2.5
            self._log.add(
                "  [BLUR]  blur_score={:.1f} => VERY BLURRY, "
                "STRONG sharpening (strength={:.2f}, sigma=2.50)".format(
                    blur_score, decisions["sharpen_strength"]
                ))
        elif blur_score < 200.0:
            decisions["sharpen_strength"] = 0.8 + (200.0 - blur_score) / 250.0
            decisions["sharpen_sigma"] = 1.5 + (200.0 - blur_score) / 300.0
            self._log.add(
                "  [BLUR]  blur_score={:.1f} => MODERATELY BLURRY, "
                "medium sharpening (strength={:.2f}, sigma={:.2f})".format(
                    blur_score, decisions["sharpen_strength"],
                    decisions["sharpen_sigma"]
                ))
        else:
            decisions["sharpen_strength"] = 0.5
            decisions["sharpen_sigma"] = 1.0
            self._log.add(
                "  [BLUR]  blur_score={:.1f} => SHARP, "
                "light sharpening only (strength=0.50, sigma=1.00)".format(blur_score)
            )

        # CLAHE decision
        if contrast < 0.15:
            decisions["clahe_clip"] = 3.0
            self._log.add(
                "  [CONTRAST] contrast={:.3f} => LOW CONTRAST, "
                "strong CLAHE (clipLimit=3.0) to restore detail visibility".format(contrast)
            )
        elif contrast > 0.35:
            decisions["clahe_clip"] = 1.0
            self._log.add(
                "  [CONTRAST] contrast={:.3f} => HIGH CONTRAST, "
                "gentle CLAHE (clipLimit=1.0) to avoid over-enhancement".format(contrast)
            )
        else:
            decisions["clahe_clip"] = self.clahe_clip
            self._log.add(
                "  [CONTRAST] contrast={:.3f} => NORMAL, "
                "standard CLAHE (clipLimit={:.1f})".format(contrast, self.clahe_clip)
            )

        # Edge-preserving decision
        if edge_density > 0.05:
            decisions["bg_blend"] = max(0.3, self.bg_blend - 0.2)
            self._log.add(
                "  [EDGES] edge_density={:.4f} => MANY EDGES, "
                "reduced bg smoothing ({:.2f}) to preserve structure".format(
                    edge_density, decisions["bg_blend"]
                ))
        else:
            decisions["bg_blend"] = self.bg_blend
            self._log.add(
                "  [EDGES] edge_density={:.4f} => NORMAL, "
                "standard bg smoothing ({:.2f})".format(
                    edge_density, decisions["bg_blend"]
                ))

        # Skin-aware decision
        if has_skin:
            skin_ratio = float(profile.get("skin_ratio", 0.0))
            self._log.add(
                "  [SKIN]  skin_ratio={:.3f} => FACE/SKIN PRESENT, "
                "all operations will be gentler on ROI to preserve natural appearance".format(
                    skin_ratio
                ))

        # Brightness warning (Pixel Layer doesn't fix this, but logs it)
        if brightness < 0.3:
            self._log.add(
                "  [LIGHT] brightness={:.3f} => LOW LIGHT detected. "
                "Pixel Layer does NOT adjust brightness (Global Layer responsibility). "
                "Proceeding with detail-only enhancement.".format(brightness)
            )
        elif brightness > 0.75:
            self._log.add(
                "  [LIGHT] brightness={:.3f} => OVEREXPOSED detected. "
                "Pixel Layer does NOT adjust exposure (Global Layer responsibility).".format(
                    brightness
                ))

        self._log.add("  Decision engine complete. {} operations planned.".format(
            sum(1 for k in ["denoise", "sharpen_strength", "clahe_clip", "bg_blend"]
                if k in decisions)
        ))

        return decisions

    # ------------------------------------------------------------------
    #  reduceNoise()  — Analysis Report: "Removes noise and artifacts"
    #                   HLD 3.2.3: "Adaptive Noise Suppression"
    # ------------------------------------------------------------------

    def reduceNoise(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        decisions: Dict,
    ) -> np.ndarray:
        """Adaptive noise suppression driven by decision engine."""
        if not decisions.get("denoise", False):
            self._log.add("reduceNoise: SKIPPED (decision engine: image is clean)")
            return img

        h_strength = float(decisions.get("denoise_h", 5.0))
        roi_weight = float(decisions.get("denoise_roi_weight", 0.4))
        bg_weight = 0.9

        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, h_strength, h_strength, 7, 21,
        )

        mask3 = _to_3ch(mask)
        blend = mask3 * roi_weight + (1.0 - mask3) * bg_weight

        out = img.astype(np.float32) * (1.0 - blend) + denoised.astype(np.float32) * blend
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)

        self._log.add(
            "reduceNoise: APPLIED (h={:.1f}, ROI_weight={:.1f}, BG_weight={:.1f})".format(
                h_strength, roi_weight, bg_weight
            )
        )
        return out

    # ------------------------------------------------------------------
    #  sharpenImage()  — Analysis Report: "Restores edges and details"
    #                    HLD 3.2.3: "Controlled Sharpening"
    #                    Project Specs: "sharpening, deblurring, detail restoration"
    # ------------------------------------------------------------------

    def sharpenImage(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        decisions: Dict,
    ) -> np.ndarray:
        """Controlled sharpening + fine-grained detail restoration.

        Sub-steps:
          a) Unsharp mask on Y channel (sharpening/deblurring)
          b) CLAHE on L channel (detail restoration)
        """
        roi_blend = float(np.clip(self.roi_blend, 0.0, 1.0))
        strength = float(decisions.get("sharpen_strength", 0.8))
        sigma = float(decisions.get("sharpen_sigma", 1.5))
        clahe_clip = float(decisions.get("clahe_clip", self.clahe_clip))

        # --- Sub-step a: Controlled Sharpening / Deblurring ---
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y = ycrcb[:, :, 0]

        y_blur = cv2.GaussianBlur(y, (0, 0), sigmaX=sigma, sigmaY=sigma)
        detail = y - y_blur
        strength_map = mask * (roi_blend * strength)

        y_sharp = y + strength_map * detail
        y_sharp = np.clip(y_sharp, 0.0, 255.0)

        ycrcb[:, :, 0] = y_sharp
        sharp_bgr = cv2.cvtColor(
            ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR
        ).astype(np.float32)

        mask3 = _to_3ch(mask)
        effective_blend = mask3 * roi_blend
        result = img.astype(np.float32) * (1.0 - effective_blend) \
            + sharp_bgr * effective_blend
        result = np.clip(result, 0.0, 255.0).astype(np.uint8)

        self._log.add(
            "sharpenImage [unsharp]: APPLIED (sigma={:.2f}, strength={:.2f}, roi_blend={:.2f})".format(
                sigma, strength, roi_blend
            )
        )

        # --- Sub-step b: Detail Restoration (CLAHE) ---
        if clahe_clip < 0.1:
            self._log.add("sharpenImage [detail]: SKIPPED (clahe_clip too low)")
            return result

        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_ch)

        enhanced_lab = cv2.merge([l_enhanced, a_ch, b_ch])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        out = result.astype(np.float32) * (1.0 - mask3) \
            + enhanced_bgr.astype(np.float32) * mask3
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)

        self._log.add(
            "sharpenImage [detail]: APPLIED (CLAHE clipLimit={:.1f}, tileGrid=8x8, ROI-only)".format(
                clahe_clip
            )
        )
        return out

    # ------------------------------------------------------------------
    #  edgePreservingFilter()  — HLD 3.2.3: "Edge-Preserving Filtering"
    # ------------------------------------------------------------------

    def edgePreservingFilter(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        decisions: Dict,
    ) -> np.ndarray:
        """Bilateral filter on background regions."""
        bg_blend = float(decisions.get("bg_blend", self.bg_blend))
        bg_blend = float(np.clip(bg_blend, 0.0, 1.0))
        bg_mask = 1.0 - mask

        if np.max(bg_mask) < 0.1 or bg_blend < 1e-3:
            self._log.add("edgePreservingFilter: SKIPPED (no background region)")
            return img

        filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

        bg_mask3 = _to_3ch(bg_mask)
        alpha = bg_mask3 * bg_blend
        out = img.astype(np.float32) * (1.0 - alpha) + filtered.astype(np.float32) * alpha
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)

        self._log.add(
            "edgePreservingFilter: APPLIED (bilateral d=9, sigmaColor=75, "
            "sigmaSpace=75, bg_blend={:.2f})".format(bg_blend)
        )
        return out

    # ------------------------------------------------------------------
    #  process()  — Full Pixel Layer pipeline (HYBRID)
    # ------------------------------------------------------------------

    def process(
        self,
        img: np.ndarray,
        profile: Dict,
        mask_map: Optional[np.ndarray] = None,
        manual_decisions: Optional[Dict] = None,
    ) -> tuple:
        """Run the full PixelBasedProcessor pipeline.

        Parameters
        ----------
        manual_decisions : dict or None
            If provided, bypasses the automatic decision engine and uses
            the user-supplied parameter values. Enables GUI manual mode.

        Sequence:
            0. Input Analysis → Decision Engine (or manual override)
            1. AI Enhancement (if trained model available) — HLD "Hybrid"
            2. reduceNoise()           — HLD 3.2.3
            3. sharpenImage()          — HLD 3.2.3
            4. edgePreservingFilter()  — HLD 3.2.3
        """
        self._log = ProcessingLog()

        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            self._log.add("[ERROR] Invalid input image")
            return img, self._log

        h, w = img.shape[:2]
        mask = _prepare_mask(mask_map, h, w)

        self._log.add(
            "PixelBasedProcessor started: {}x{} | roi_blend={:.2f}, "
            "bg_blend={:.2f}, clahe_clip={:.1f}".format(
                w, h, self.roi_blend, self.bg_blend, self.clahe_clip
            )
        )

        # Step 0: Get decisions (auto or manual)
        if manual_decisions is not None:
            decisions = manual_decisions
            self._log.add("--- Mode: MANUAL (user-controlled) ---")
            self._log.add("  Sharpening:      strength={:.2f}, sigma={:.2f}".format(
                decisions.get("sharpen_strength", 0.5),
                decisions.get("sharpen_sigma", 1.5)))
            if decisions.get("denoise", False):
                self._log.add("  Noise Reduction: ON (h={:.1f}, roi_weight={:.2f})".format(
                    decisions.get("denoise_h", 5.0),
                    decisions.get("denoise_roi_weight", 0.4)))
            else:
                self._log.add("  Noise Reduction: OFF")
            self._log.add("  Detail (CLAHE):  clipLimit={:.1f}".format(
                decisions.get("clahe_clip", 2.0)))
            self._log.add("  BG Smoothing:    blend={:.2f}".format(
                decisions.get("bg_blend", 0.5)))
        else:
            decisions = self._analyze_and_decide(profile)

        # Step 1: AI Enhancement (Hybrid — HLD 3.1, Poster)
        self._log.add("--- AI Enhancement (Hybrid) ---")
        model, device = _load_enhance_net(self.checkpoint_path, self._log)
        if model is not None:
            result = _ai_enhance(img, model, device, mask, self.roi_blend, self._log)
        else:
            self._log.add("AI Enhance: SKIPPED — using classical pipeline")
            result = img.copy()

        # Step 2: reduceNoise() — HLD 3.2.3 "Adaptive Noise Suppression"
        self._log.add("--- Classical Pipeline ---")
        result = self.reduceNoise(result, mask, decisions)

        # Step 3: sharpenImage() — HLD 3.2.3 "Controlled Sharpening"
        result = self.sharpenImage(result, mask, decisions)

        # Step 4: edgePreservingFilter() — HLD 3.2.3 "Edge-Preserving Filtering"
        result = self.edgePreservingFilter(result, mask, decisions)

        self._log.add("PixelBasedProcessor completed successfully.")
        return result, self._log
