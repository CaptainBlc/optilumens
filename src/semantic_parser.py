"""
Face Parser — Semantic Layer (Stage 1)

CMPE 491 Senior Design Project.

Per-pixel face region labeling using ParseNet (CelebAMask-HQ pretrained)
via facexlib. Detects faces, parses each into 19 region classes, and
projects masks back to original image coordinates.

ParseNet chosen over BiSeNet to match facexlib FaceRestoreHelper's
internal default (better lips/mouth/hair recall).

Region labels (CelebAMask-HQ):
    0 background  1 skin       2 l_brow    3 r_brow
    4 l_eye       5 r_eye      6 eye_g     7 l_ear
    8 r_ear       9 ear_r     10 nose     11 mouth
   12 u_lip      13 l_lip     14 neck     15 neck_l
   16 cloth      17 hair      18 hat

NOT integrated into pipeline.py yet — exercised standalone via __main__.
Stage 2 (region_enhancer) consumes SemanticResult to apply per-region
processing; Stage 3 wires both into ImageEnhancementPipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    pass

# Tri-state probe — None until first call, then True/False
_FACEXLIB_AVAILABLE: Optional[bool] = None


# ── Label table (CelebAMask-HQ) ───────────────────────────────────────

LABELS: Dict[int, str] = {
    0:  "background",  1:  "skin",      2:  "l_brow",   3:  "r_brow",
    4:  "l_eye",       5:  "r_eye",     6:  "eye_g",    7:  "l_ear",
    8:  "r_ear",       9:  "ear_r",    10:  "nose",    11:  "mouth",
    12: "u_lip",      13: "l_lip",     14: "neck",     15: "neck_l",
    16: "cloth",      17: "hair",      18: "hat",
}

# Visualization palette (BGR for OpenCV).
PALETTE: Dict[int, Tuple[int, int, int]] = {
    0:  (  0,   0,   0),   # background
    1:  (180, 200, 220),   # skin
    2:  ( 30,  70, 140),   # l_brow
    3:  ( 30,  70, 140),   # r_brow
    4:  ( 80, 220,  90),   # l_eye
    5:  ( 80, 220,  90),   # r_eye
    6:  (180, 180,  30),   # eye_g
    7:  (200, 150, 100),   # l_ear
    8:  (200, 150, 100),   # r_ear
    9:  (220, 220,  60),   # ear_r
    10: (160, 100, 230),   # nose
    11: ( 40,  40, 220),   # mouth
    12: ( 80,  80, 240),   # u_lip
    13: ( 80,  80, 240),   # l_lip
    14: (200, 200, 200),   # neck
    15: ( 50, 200, 200),   # neck_l
    16: ( 90,  60,  30),   # cloth
    17: ( 60,  60,  60),   # hair
    18: ( 80,  20, 140),   # hat
}


# ── Result dataclasses ────────────────────────────────────────────────

@dataclass
class FaceRegion:
    """One labeled region within a face, in original image coordinates."""
    label_id: int
    label_name: str
    mask: np.ndarray                       # uint8, 0/255, original size
    bbox: Tuple[int, int, int, int]        # x, y, w, h
    pixel_count: int


@dataclass
class FaceParseResult:
    """Parsing output for a single detected face."""
    face_idx: int
    face_bbox: Tuple[int, int, int, int]   # x, y, w, h in original coords
    label_map: np.ndarray                  # uint8, 19-class, original size
    regions: Dict[str, FaceRegion] = field(default_factory=dict)


@dataclass
class SemanticResult:
    """Aggregate output across all detected faces."""
    faces: List[FaceParseResult] = field(default_factory=list)
    label_map: Optional[np.ndarray] = None  # combined 19-class map
    log: List[str] = field(default_factory=list)
    success: bool = False


# ── Helpers ───────────────────────────────────────────────────────────

def _check_facexlib() -> bool:
    """Lazy probe — facexlib pulls in DLLs that can crash at import time."""
    global _FACEXLIB_AVAILABLE
    if _FACEXLIB_AVAILABLE is None:
        try:
            from facexlib.parsing import init_parsing_model  # noqa: F401
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper  # noqa: F401
            _FACEXLIB_AVAILABLE = True
        except (ImportError, OSError):
            _FACEXLIB_AVAILABLE = False
    return _FACEXLIB_AVAILABLE


def _face_to_tensor(face_bgr: np.ndarray, device) -> "torch.Tensor":
    """BGR uint8 [512,512,3] → normalized RGB tensor [1,3,512,512] in [-1, 1].

    Matches facexlib/GFPGAN preprocessing: img/255, then normalize with
    mean=0.5, std=0.5 (equivalent to img/255 * 2 - 1).
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return t * 2.0 - 1.0


# ── Main class ────────────────────────────────────────────────────────

class FaceParser:
    """
    ParseNet-based face parser. Detects faces with RetinaFace, runs 19-class
    parsing on each aligned 512×512 crop, projects masks back to the
    original image.

    Usage:
        parser = FaceParser()
        result = parser.parse(image_bgr)
        for face in result.faces:
            skin_region = face.regions.get("skin")   # FaceRegion or None

    First run downloads ParseNet (~80 MB) and RetinaFace (~104 MB) weights
    into facexlib's weights cache. Subsequent runs are offline.
    """

    FACE_SIZE = 512

    def __init__(self, device: Optional[str] = None) -> None:
        self._model = None
        self._helper = None
        self._log: List[str] = []

        if _TORCH_AVAILABLE:
            if device is None:
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(device)
        else:
            self._device = None

    # ── lazy init ─────────────────────────────────────────────────────

    def _init_model(self) -> bool:
        if self._model is not None:
            return True
        if not _TORCH_AVAILABLE:
            self._log.append("[ERROR] PyTorch not available")
            return False
        if not _check_facexlib():
            self._log.append("[ERROR] facexlib not available")
            return False

        try:
            from facexlib.parsing import init_parsing_model
            self._model = init_parsing_model(
                model_name="bisenet",
                device=self._device,
            )
            self._model.eval()
            self._log.append(
                "FaceParser: BiSeNet LOADED (device={})".format(self._device))
        except Exception as e:
            self._log.append("[ERROR] BiSeNet load failed: {}".format(e))
            self._model = None
            return False

        try:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            self._helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=self.FACE_SIZE,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=False,             # we call BiSeNet ourselves
                device=self._device,
            )
            self._log.append("FaceParser: detector READY (RetinaFace)")
        except Exception as e:
            self._log.append("[ERROR] Detector init failed: {}".format(e))
            self._helper = None
            return False

        return True

    # ── Public API ────────────────────────────────────────────────────

    def parse(self, img: np.ndarray) -> SemanticResult:
        """
        Detect faces, parse each into 19 region classes, project masks
        back to original image coordinates.
        """
        self._log = []
        result = SemanticResult()

        if img is None or img.size == 0:
            self._log.append("[ERROR] Invalid input")
            result.log = self._log
            return result

        if not self._init_model():
            result.log = self._log
            return result

        H, W = img.shape[:2]

        try:
            self._helper.clean_all()
            self._helper.read_image(img)
            self._helper.get_face_landmarks_5(eye_dist_threshold=5)
            self._helper.align_warp_face()
            n_faces = len(self._helper.cropped_faces)
            self._log.append("FaceParser: detected {} face(s)".format(n_faces))
        except Exception as e:
            self._log.append("[ERROR] Detection failed: {}".format(e))
            result.log = self._log
            return result

        if n_faces == 0:
            result.success = True
            result.log = self._log
            return result

        try:
            self._helper.get_inverse_affine(None)
        except Exception as e:
            self._log.append("[ERROR] Inverse affine failed: {}".format(e))
            result.log = self._log
            return result

        combined_map = np.zeros((H, W), dtype=np.uint8)

        for i, face in enumerate(self._helper.cropped_faces):
            label_512 = self._parse_aligned(face)
            if label_512 is None:
                continue

            inv = (self._helper.inverse_affine_matrices[i]
                   if i < len(self._helper.inverse_affine_matrices) else None)
            if inv is None:
                continue

            # INTER_NEAREST: never interpolate across class boundaries
            label_full = cv2.warpAffine(
                label_512, inv, (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            ys, xs = np.where(label_full > 0)
            if ys.size == 0:
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
            face_bbox = (x0, y0, x1 - x0, y1 - y0)

            face_res = FaceParseResult(
                face_idx=i,
                face_bbox=face_bbox,
                label_map=label_full,
            )

            for lid, lname in LABELS.items():
                if lid == 0:
                    continue
                m = (label_full == lid)
                count = int(m.sum())
                if count == 0:
                    continue
                ry, rx = np.where(m)
                rx0, ry0 = int(rx.min()), int(ry.min())
                rx1, ry1 = int(rx.max()) + 1, int(ry.max()) + 1
                face_res.regions[lname] = FaceRegion(
                    label_id=lid,
                    label_name=lname,
                    mask=(m.astype(np.uint8) * 255),
                    bbox=(rx0, ry0, rx1 - rx0, ry1 - ry0),
                    pixel_count=count,
                )

            self._log.append("  Face {}: {} region(s)".format(
                i + 1, len(face_res.regions)))

            result.faces.append(face_res)
            np.maximum(combined_map, label_full, out=combined_map)

        result.label_map = combined_map
        result.success = True
        result.log = self._log
        return result

    def parse_face_crop(self, face_512_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Parse a pre-aligned 512×512 face crop. Returns uint8 [512,512]
        label map, or None on error.

        Useful when callers already aligned faces (e.g. integrating with
        FaceRestorer's existing crops to skip re-detection).
        """
        self._log = []
        if not self._init_model():
            return None
        return self._parse_aligned(face_512_bgr)

    # ── internal ──────────────────────────────────────────────────────

    def _parse_aligned(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run ParseNet on a 512×512 face. Returns uint8 label map [512,512]."""
        if face_bgr.shape[0] != self.FACE_SIZE or face_bgr.shape[1] != self.FACE_SIZE:
            face_bgr = cv2.resize(face_bgr, (self.FACE_SIZE, self.FACE_SIZE))

        try:
            with torch.no_grad():
                t = _face_to_tensor(face_bgr, self._device)
                out = self._model(t)[0]              # [1, 19, 512, 512]
                parse_map = out.argmax(dim=1).squeeze(0).cpu().numpy()
            return parse_map.astype(np.uint8)
        except Exception as e:
            self._log.append("[ERROR] Parsing failed: {}".format(e))
            return None

    # ── Visualization utilities ───────────────────────────────────────

    @staticmethod
    def colorize(label_map: np.ndarray) -> np.ndarray:
        """Render a 19-class label map as a BGR image using PALETTE."""
        h, w = label_map.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for lid, color in PALETTE.items():
            out[label_map == lid] = color
        return out

    @staticmethod
    def overlay(img: np.ndarray, label_map: np.ndarray,
                alpha: float = 0.5) -> np.ndarray:
        """Blend the colorized label map over the original image."""
        c = FaceParser.colorize(label_map)
        mask = (label_map > 0).astype(np.float32) * alpha
        m3 = cv2.merge([mask, mask, mask])
        return (img.astype(np.float32) * (1 - m3)
                + c.astype(np.float32) * m3).astype(np.uint8)

    @staticmethod
    def draw_bboxes(img: np.ndarray, result: SemanticResult,
                    region_names: Optional[List[str]] = None) -> np.ndarray:
        """Draw face + per-region bboxes with labels. Returns annotated copy."""
        out = img.copy()
        for face in result.faces:
            x, y, w, h = face.face_bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(out, "Face {}".format(face.face_idx + 1),
                        (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                        cv2.LINE_AA)
            for name, r in face.regions.items():
                if region_names is not None and name not in region_names:
                    continue
                rx, ry, rw, rh = r.bbox
                color = PALETTE.get(r.label_id, (255, 255, 255))
                cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), color, 1)
                cv2.putText(out, name, (rx, max(0, ry - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                            cv2.LINE_AA)
        return out

    def get_log(self) -> List[str]:
        return self._log.copy()


# ── Standalone smoke test ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import sys

    ap = argparse.ArgumentParser(
        description="Stage 1 smoke test: face parser")
    ap.add_argument("input", help="input image path")
    ap.add_argument("--out", default=None,
                    help="output path (default: <input>_parsed.png)")
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="overlay opacity (0..1)")
    ap.add_argument("--bboxes", action="store_true",
                    help="also draw region bounding boxes")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print("[ERROR] Cannot read: {}".format(args.input))
        sys.exit(1)

    parser = FaceParser()
    result = parser.parse(img)
    for line in result.log:
        print(line)

    if not result.success:
        sys.exit(2)

    if not result.faces:
        print("(no faces detected)")
        sys.exit(0)

    color = FaceParser.colorize(result.label_map)
    over  = FaceParser.overlay(img, result.label_map, alpha=args.alpha)
    if args.bboxes:
        over = FaceParser.draw_bboxes(over, result)
    sbs = cv2.hconcat([img, color, over])

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(args.input)[0]
        out_path = base + "_parsed.png"
    cv2.imwrite(out_path, sbs)
    print("\nSaved: {}".format(out_path))

    for face in result.faces:
        print("\nFace {}  bbox={}  regions:".format(
            face.face_idx + 1, face.face_bbox))
        for name, r in sorted(face.regions.items(),
                              key=lambda kv: -kv[1].pixel_count):
            print("  {:10s} px={:>7d}  bbox={}".format(
                name, r.pixel_count, r.bbox))
