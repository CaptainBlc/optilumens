"""
Batch Processor — background multi-image enhancement.

Implements Analysis Report §3.5.1 scenarios #1 and #2 ("offline editing"
and "batch processing") — processes every image in a folder through the
standard ImageEnhancementPipeline without blocking the GUI.

Design:
    - Runs inside a QThread spawned by the GUI.
    - Emits per-file progress signals for live status updates.
    - Cancellable mid-run: checks a flag between iterations.
    - Writes enhanced outputs next to originals under outputs/batch_<ts>/.

Contract
--------
    worker = BatchWorker(file_list, out_dir, fidelity=0.5, only_center=False)
    worker.progress.connect(...)     # (index, total, filename, status, elapsed_s)
    worker.finished_all.connect(...) # (success_count, failed_count, summary)
    worker.start()
    worker.cancel()  # graceful stop between images
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
from PyQt6.QtCore import QThread, pyqtSignal


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass
class BatchItemResult:
    """One row of the batch summary table."""
    index:      int
    filename:   str
    status:     str            # "ok" | "failed" | "skipped" | "cancelled"
    out_path:   Optional[str] = None
    elapsed_s:  float = 0.0
    error:      Optional[str] = None
    profile:    dict = field(default_factory=dict)
    metrics:    dict = field(default_factory=dict)


def collect_images(folder: str, recursive: bool = False) -> List[str]:
    """Return sorted absolute paths of supported images in `folder`."""
    out: List[str] = []
    if not folder or not os.path.isdir(folder):
        return out
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                    out.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(folder)):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                out.append(p)
    return sorted(out)


class BatchWorker(QThread):
    """
    Runs ImageEnhancementPipeline across a list of files in a background thread.

    Signals
    -------
    progress(int idx, int total, str filename, str status, float elapsed):
        Emitted at the *start* ("processing") and *end* ("ok"/"failed")
        of each file.
    finished_all(int ok_count, int fail_count, list results):
        Emitted once the entire queue is done (or cancelled).
    log_line(str):
        Free-form log messages for the GUI console.
    """

    progress     = pyqtSignal(int, int, str, str, float)
    finished_all = pyqtSignal(int, int, list)
    log_line     = pyqtSignal(str)

    def __init__(self,
                 files: List[str],
                 out_dir: str,
                 fidelity: float = 0.5,
                 only_center: bool = False) -> None:
        super().__init__()
        self._files       = list(files)
        self._out_dir     = out_dir
        self._fidelity    = float(fidelity)
        self._only_center = bool(only_center)
        self._cancel      = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        # Lazy-import so the module is cheap to load (and avoids torch-
        # related DLL issues surfacing at GUI startup).
        from pipeline import ImageEnhancementPipeline

        os.makedirs(self._out_dir, exist_ok=True)
        pipe = ImageEnhancementPipeline(fidelity_weight=self._fidelity)

        results: List[BatchItemResult] = []
        ok = fail = 0
        total = len(self._files)
        self.log_line.emit(
            "=== Batch start: {} files | out={} ===".format(total, self._out_dir))

        for i, path in enumerate(self._files):
            if self._cancel:
                self.log_line.emit("[CANCELLED] stopping batch at {}/{}".format(i, total))
                results.append(BatchItemResult(
                    index=i, filename=os.path.basename(path), status="cancelled"))
                break

            fname = os.path.basename(path)
            self.progress.emit(i, total, fname, "processing", 0.0)
            t0 = time.time()

            try:
                img = cv2.imread(path)
                if img is None:
                    raise RuntimeError("cv2.imread returned None (unsupported or corrupt)")

                res = pipe.restoreImage(
                    img,
                    fidelity_weight=self._fidelity,
                    only_center_face=self._only_center,
                )

                if res is None or not res.success or res.restored is None:
                    raise RuntimeError("pipeline reported failure")

                stem, ext = os.path.splitext(fname)
                if ext.lower() not in SUPPORTED_EXT:
                    ext = ".jpg"
                out_path = os.path.join(self._out_dir, stem + "_enhanced" + ext)
                cv2.imwrite(out_path, res.restored)

                dt = time.time() - t0
                item = BatchItemResult(
                    index=i, filename=fname, status="ok",
                    out_path=out_path, elapsed_s=dt,
                )
                if res.profile is not None:
                    item.profile = {
                        "brightness": res.profile.brightness,
                        "blur":       res.profile.blur_score,
                        "skin":       res.profile.skin_ratio,
                    }
                if res.metrics is not None:
                    item.metrics = {
                        "psnr": res.metrics.psnr,
                        "ssim": res.metrics.ssim,
                    }
                results.append(item)
                ok += 1
                self.progress.emit(i, total, fname, "ok", dt)
                self.log_line.emit(
                    "  [{}/{}] OK   {:.1f}s   {}".format(i + 1, total, dt, fname))

            except Exception as e:
                dt = time.time() - t0
                err = str(e)
                results.append(BatchItemResult(
                    index=i, filename=fname, status="failed",
                    elapsed_s=dt, error=err,
                ))
                fail += 1
                self.progress.emit(i, total, fname, "failed", dt)
                self.log_line.emit(
                    "  [{}/{}] FAIL {}: {}".format(i + 1, total, fname, err))
                self.log_line.emit(traceback.format_exc(limit=1).rstrip())

        self.log_line.emit(
            "=== Batch done: {} ok, {} failed ===".format(ok, fail))
        self.finished_all.emit(ok, fail, results)
