"""
GFPGAN Enhancement System — GUI
CMPE 491 Senior Design Project.
"""

import os
import sys
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QCheckBox, QTextEdit, QLineEdit, QFrame,
    QFileDialog, QSplitter, QApplication, QSizePolicy,
    QGridLayout, QButtonGroup, QDialog, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, QRect, QPoint, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QPixmap, QImage, QAction, QKeySequence,
    QPainter, QPen, QColor, QFont, QBrush, QRegion,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import ImageEnhancementPipeline
from profiler import ImageProfiler
from camera_capture import CameraCapture
from batch_processor import BatchWorker, collect_images
from chat_commands import parse as chat_parse, ChatExecutor
from live_filters import make_filter, BaseFilter

RIGHT_W     = 265
SLD_LBL_W   = 80
SLD_VAL_W   = 40
CARD_PAD    = 8

STYLE = """
* { font-family: "Segoe UI", Arial; font-size: 11px; }
QMainWindow { background: #0d0d0d; }
QMenuBar { background: #151515; color: #aaa; border-bottom: 1px solid #2a2a2a; }
QMenuBar::item:selected { background: #222; }
QMenu { background: #1a1a1a; border: 1px solid #333; color: #ddd; }
QMenu::item:selected { background: #c53030; }

QFrame#toolbar {
    background: #151515;
    border-bottom: 2px solid #c53030;
    min-height: 38px; max-height: 38px;
}

QPushButton {
    background: #252525; color: #ddd; border: 1px solid #333;
    border-radius: 4px; padding: 4px 11px;
}
QPushButton:hover { background: #2e2e2e; border-color: #c53030; }
QPushButton:pressed { background: #c53030; color: #fff; }
QPushButton:disabled { background: #181818; color: #444; border-color: #222; }
QPushButton#accent {
    background: #c53030; color: #fff; border: none;
    font-weight: bold; padding: 4px 18px; border-radius: 5px;
}
QPushButton#accent:hover { background: #dc4040; }
QPushButton#accent:disabled { background: #333; color: #555; }
QPushButton#vb {
    background: none; color: #666; border: 1px solid #2a2a2a;
    border-radius: 3px; padding: 3px 9px; font-size: 10px;
}
QPushButton#vb:checked { background: #c53030; color: #fff; border-color: #c53030; }
QPushButton#vb:hover   { border-color: #c53030; color: #eee; }

QFrame#card {
    background: #111; border: 1px solid #2a2a2a; border-radius: 6px;
}
QSlider::groove:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #252525,stop:1 #c53030);
    height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #eee; width: 12px; height: 12px;
    margin: -4px 0; border-radius: 6px; border: 2px solid #c53030;
}
QSlider::handle:horizontal:disabled { background: #444; border-color: #333; }
QCheckBox { color: #999; spacing: 4px; }
QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #333; background: #1a1a1a; }
QCheckBox::indicator:checked { background: #c53030; border-color: #c53030; }

QTextEdit#log {
    background: #0a0a0a; color: #b0ffb0;
    font-family: Consolas, monospace; font-size: 10px;
    border: 1px solid #2a2a2a; border-radius: 4px; padding: 3px;
}
QScrollBar:vertical { background: #0d0d0d; width: 6px; }
QScrollBar::handle:vertical { background: #2e2e2e; min-height: 20px; border-radius: 3px; }
QScrollBar::handle:vertical:hover { background: #c53030; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QLabel { color: #ddd; }
"""


# ── Worker thread (prevents GUI freeze during processing) ──────────

class EnhanceWorker(QThread):
    done = pyqtSignal(object)

    def __init__(self, pipeline, image, fidelity, center_only):
        super().__init__()
        self._pipe = pipeline
        self._img = image
        self._fw = fidelity
        self._center = center_only

    def run(self):
        result = self._pipe.restoreImage(
            self._img,
            fidelity_weight=self._fw,
            only_center_face=self._center,
        )
        self.done.emit(result)


# ── Swipe compare widget ───────────────────────────────────────────

class SwipeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._bef = self._aft = None
        self._ratio = 0.5
        self._drag = False
        self._ir = QRect()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_images(self, bef, aft):
        self._bef, self._aft = bef, aft
        self.update()

    def clear(self):
        self._bef = self._aft = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor("#0e0e0e"))

        if not self._bef or not self._aft:
            p.setPen(QColor("#444"))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Enhance an image first, then click Compare")
            p.end(); return

        sb = self._bef.scaled(W, H, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        sa = self._aft.scaled(W, H, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        iw, ih = sb.width(), sb.height()
        xo, yo = (W - iw) // 2, (H - ih) // 2
        self._ir = QRect(xo, yo, iw, ih)
        sx = int(iw * self._ratio)
        lx = xo + sx

        p.drawPixmap(xo, yo, sa)
        p.setClipRegion(QRegion(QRect(xo, yo, sx, ih)))
        p.drawPixmap(xo, yo, sb)
        p.setClipping(False)

        p.setPen(QPen(QColor("#c53030"), 2))
        p.drawLine(lx, yo, lx, yo + ih)

        cy = yo + ih // 2
        p.setBrush(QBrush(QColor("#c53030")))
        p.setPen(QPen(QColor("#fff"), 1))
        p.drawEllipse(QPoint(lx, cy), 12, 12)
        p.setPen(QPen(QColor("#fff"), 2))
        for dx, d in [(-5, -1), (5, 1)]:
            p.drawLine(lx + dx, cy, lx + dx + d * 3, cy - 3)
            p.drawLine(lx + dx, cy, lx + dx + d * 3, cy + 3)

        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.setPen(QColor("#ffaa33"))
        p.drawText(xo + 6, yo + 15, "ORIGINAL")
        p.setPen(QColor("#33ff99"))
        p.drawText(xo + iw - 72, yo + 15, "ENHANCED")

        pct = int(self._ratio * 100)
        p.setFont(QFont("Segoe UI", 8))
        p.setBrush(QBrush(QColor(0, 0, 0, 180)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(lx - 18, yo + ih - 20, 36, 14, 3, 3)
        p.setPen(QColor("#fff"))
        p.drawText(QRect(lx - 18, yo + ih - 20, 36, 14),
                   Qt.AlignmentFlag.AlignCenter, "{}%".format(pct))
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._ir.contains(e.pos()):
            self._drag = True; self._upd(e.pos())

    def mouseMoveEvent(self, e):
        self.setCursor(Qt.CursorShape.SplitHCursor
                       if self._ir.contains(e.pos())
                       else Qt.CursorShape.ArrowCursor)
        if self._drag: self._upd(e.pos())

    def mouseReleaseEvent(self, _): self._drag = False

    def _upd(self, pos):
        if self._ir.width() > 0:
            self._ratio = max(0.02, min(0.98,
                (pos.x() - self._ir.x()) / self._ir.width()))
            self.update()


# ── Simple viewport ───────────────────────────────────────────────

class Viewport(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._px = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background:#0e0e0e; border:1px solid #2a2a2a;"
            "border-radius:6px; color:#555;")

    def img(self, pm):
        self._px = pm; self._fit()

    def clr(self):
        self._px = None; super().setPixmap(QPixmap())

    def _fit(self):
        if self._px and not self._px.isNull():
            super().setPixmap(self._px.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e); self._fit()


# ── Metric tile ───────────────────────────────────────────────────

def _tile(label, accent=False):
    f = QFrame()
    f.setFixedHeight(44)
    f.setStyleSheet("background:{}; border-radius:4px;".format(
        "#3a0a0a" if accent else "#181818"))
    l = QVBoxLayout(f)
    l.setContentsMargins(4, 2, 4, 2); l.setSpacing(0)
    v = QLabel("--")
    v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v.setStyleSheet("font-size:13px; font-weight:bold; color:#eee;")
    n = QLabel(label)
    n.setAlignment(Qt.AlignmentFlag.AlignCenter)
    n.setStyleSheet("font-size:9px; color:#555;")
    l.addWidget(v); l.addWidget(n)
    return f, v


# ── BatchDialog (Sprint 2) ────────────────────────────────────────

class BatchDialog(QDialog):
    """Non-blocking batch-processing dialog with per-file status table."""

    def __init__(self, files, out_dir: str, fidelity: float,
                 only_center: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Enhancement — OptiLumen")
        self.setMinimumSize(720, 480)
        self.setModal(False)

        self._files    = list(files)
        self._out_dir  = out_dir
        self._total    = len(self._files)
        self._ok = self._fail = 0
        self._done = False

        lay = QVBoxLayout(self); lay.setContentsMargins(12, 10, 12, 10); lay.setSpacing(8)

        hdr = QLabel(
            "Processing <b>{}</b> image(s)<br/>"
            "<span style='color:#888'>Output: {}</span>".format(
                self._total, out_dir))
        hdr.setStyleSheet("color:#ddd; font-size:12px;")
        lay.addWidget(hdr)

        self._bar = QProgressBar(); self._bar.setRange(0, self._total)
        self._bar.setFormat("%v / %m  (%p%)")
        self._bar.setStyleSheet(
            "QProgressBar{background:#1a1a1a;border:1px solid #333;border-radius:4px;"
            "color:#ddd;text-align:center;height:18px;}"
            "QProgressBar::chunk{background:#2a7a2a;border-radius:3px;}")
        lay.addWidget(self._bar)

        self._cur = QLabel("Queued…"); self._cur.setStyleSheet("color:#aaa; font-size:11px;")
        lay.addWidget(self._cur)

        self._tbl = QTableWidget(self._total, 4)
        self._tbl.setHorizontalHeaderLabels(["#", "File", "Status", "Time"])
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._tbl.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._tbl.setStyleSheet(
            "QTableWidget{background:#111;color:#ccc;gridline-color:#222;"
            "border:1px solid #2a2a2a;font-size:11px;}"
            "QHeaderView::section{background:#1a1a1a;color:#aaa;border:none;"
            "padding:4px;border-bottom:1px solid #333;}")
        for i, p in enumerate(self._files):
            self._tbl.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._tbl.setItem(i, 1, QTableWidgetItem(os.path.basename(p)))
            self._tbl.setItem(i, 2, QTableWidgetItem("—"))
            self._tbl.setItem(i, 3, QTableWidgetItem(""))
        lay.addWidget(self._tbl, stretch=1)

        self._log = QLabel(""); self._log.setStyleSheet(
            "color:#8a8a8a; font-family:Consolas; font-size:10px;")
        self._log.setWordWrap(True); self._log.setMinimumHeight(32)
        lay.addWidget(self._log)

        br = QHBoxLayout(); br.addStretch()
        self._btnCancel = QPushButton("Cancel"); self._btnCancel.clicked.connect(self._cancel)
        self._btnClose  = QPushButton("Close");  self._btnClose.setEnabled(False)
        self._btnClose.clicked.connect(self.accept)
        self._btnOpen   = QPushButton("Open Output Folder"); self._btnOpen.setEnabled(False)
        self._btnOpen.clicked.connect(self._open_out)
        br.addWidget(self._btnOpen); br.addWidget(self._btnCancel); br.addWidget(self._btnClose)
        lay.addLayout(br)

        self._worker = BatchWorker(self._files, out_dir, fidelity, only_center)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log)
        self._worker.finished_all.connect(self._on_finish)
        self._worker.start()

    # ── slots ──
    def _on_progress(self, idx, total, fname, status, elapsed):
        self._tbl.setItem(idx, 2, QTableWidgetItem(status.upper()))
        it_status = self._tbl.item(idx, 2)
        color = {"processing": "#ffb84d", "ok": "#7ed87e",
                 "failed": "#e06666", "cancelled": "#888"}.get(status, "#aaa")
        it_status.setForeground(QBrush(QColor(color)))

        if status == "processing":
            self._cur.setText("▶ [{}/{}] {}".format(idx + 1, total, fname))
        else:
            self._tbl.setItem(idx, 3, QTableWidgetItem("{:.1f}s".format(elapsed)))
            self._bar.setValue(idx + 1)
        self._tbl.scrollToItem(self._tbl.item(idx, 0))

    def _on_log(self, line: str):
        self._log.setText(line[-160:])

    def _on_finish(self, ok: int, fail: int, _results):
        self._ok, self._fail = ok, fail
        self._done = True
        self._btnCancel.setEnabled(False)
        self._btnClose.setEnabled(True)
        self._btnOpen.setEnabled(ok > 0)
        tot = ok + fail
        if tot == 0:
            self._cur.setText("No files processed.")
        else:
            self._cur.setText(
                "<b>Done.</b>  {} ok · {} failed  ({:.0f}%)".format(
                    ok, fail, 100.0 * ok / max(1, tot)))

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._cur.setText("Cancelling…")
            self._btnCancel.setEnabled(False)

    def _open_out(self):
        try:
            if sys.platform == "win32":
                os.startfile(self._out_dir)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system('open "{}"'.format(self._out_dir))
            else:
                os.system('xdg-open "{}"'.format(self._out_dir))
        except Exception:
            pass

    def closeEvent(self, e):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(e)


# ── MainWindow ────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    V_ORIG = 0; V_REST = 1; V_CMP = 2; V_DIF = 3; V_LIVE = 4

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GFPGAN Enhancement System — OptiLumen CMPE491")
        self.setMinimumSize(960, 620)
        self.resize(1300, 780)
        self.setAcceptDrops(True)

        self._orig = self._rest = self._dif = None
        self._pipe = ImageEnhancementPipeline()
        self._prof = ImageProfiler()
        self._view = self.V_ORIG
        self._worker = None

        # Live camera state (Analysis Report Scenario 4)
        self._cam = CameraCapture()
        self._cam_timer = QTimer(self)
        self._cam_timer.setInterval(33)  # ~30 FPS
        self._cam_timer.timeout.connect(self._cam_tick)
        self._cam_frame: Optional[np.ndarray] = None

        # Live filter (Sprint 4 — Analysis §3.5.1 #4 ext.)
        self._filter: BaseFilter = make_filter("OFF")
        self._filter_name: str = "OFF"

        self._build_menu()
        self._build_ui()
        self.setStyleSheet(STYLE)

    def closeEvent(self, e):
        try:
            self._cam_timer.stop()
            self._cam.close()
        except Exception:
            pass
        super().closeEvent(e)

    # ── menu ──────────────────────────────────────────────────────

    def _build_menu(self):
        fm = self.menuBar().addMenu("File")
        for t, sc, fn in [("Open", "Ctrl+O", self._load),
                           ("Save", "Ctrl+S", self._save),
                           ("sep", "", None),
                           ("Quit", "Ctrl+Q", self.close)]:
            if t == "sep":
                fm.addSeparator()
            else:
                a = QAction(t, self); a.setShortcut(QKeySequence(sc))
                a.triggered.connect(fn); fm.addAction(a)

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)

        # toolbar
        tb = QFrame(); tb.setObjectName("toolbar")
        tl = QHBoxLayout(tb); tl.setContentsMargins(8, 3, 8, 3); tl.setSpacing(5)

        self._bLoad = QPushButton("Open"); self._bLoad.clicked.connect(self._load)
        self._bSave = QPushButton("Save"); self._bSave.setEnabled(False)
        self._bSave.clicked.connect(self._save)
        self._bRest = QPushButton("Restore Face(s)"); self._bRest.setObjectName("accent")
        self._bRest.setEnabled(False); self._bRest.clicked.connect(self._restore)
        self._bRset = QPushButton("Reset"); self._bRset.setEnabled(False)
        self._bRset.clicked.connect(self._reset)
        self._bBatch = QPushButton("Batch…"); self._bBatch.clicked.connect(self._batch)

        for b in (self._bLoad, self._bSave, self._bRest, self._bRset, self._bBatch):
            tl.addWidget(b)
        tl.addSpacing(8)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#2a2a2a;"); tl.addWidget(sep); tl.addSpacing(4)

        # Live-camera toggle (Analysis Report Scenario 4)
        self._bLive = QPushButton("● Live"); self._bLive.setCheckable(True)
        self._bLive.setStyleSheet(
            "QPushButton:checked { background:#2a7a2a; color:#fff;"
            " border-color:#2a7a2a; font-weight:bold; }")
        self._bLive.clicked.connect(self._toggle_live)
        tl.addWidget(self._bLive)

        self._bShot = QPushButton("Capture"); self._bShot.setVisible(False)
        self._bShot.clicked.connect(self._capture_shot)
        tl.addWidget(self._bShot)

        # Live filter selector (Sprint 4) — visible only in live mode
        self._filterLbl = QLabel("Filter:"); self._filterLbl.setVisible(False)
        self._filterLbl.setStyleSheet("color:#888; font-size:10px; padding:0 3px;")
        tl.addWidget(self._filterLbl)

        self._fbtns = []
        self._fgrp  = QButtonGroup(self); self._fgrp.setExclusive(True)
        for i, nm in enumerate(["OFF", "BEAUTY", "ENHANCE", "AI"]):
            b = QPushButton(nm); b.setObjectName("vb"); b.setCheckable(True)
            b.setVisible(False); b.setMinimumWidth(62)
            if i == 0: b.setChecked(True)
            self._fgrp.addButton(b, i); self._fbtns.append(b); tl.addWidget(b)
        self._fgrp.idClicked.connect(self._set_filter)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color:#2a2a2a;"); tl.addWidget(sep2); tl.addSpacing(4)

        self._vbtns = []; self._vgrp = QButtonGroup(self); self._vgrp.setExclusive(True)
        for i, nm in enumerate(["Original", "Restored", "Compare", "Diff"]):
            b = QPushButton(nm); b.setObjectName("vb"); b.setCheckable(True)
            if i == 0: b.setChecked(True)
            self._vgrp.addButton(b, i); self._vbtns.append(b); tl.addWidget(b)
        self._vgrp.idClicked.connect(self._chg_view)
        tl.addStretch()

        self._ckCenter = QCheckBox("Center face only")
        tl.addWidget(self._ckCenter)
        rl.addWidget(tb)

        # body
        vsp = QSplitter(Qt.Orientation.Vertical); vsp.setHandleWidth(2)
        hsp = QSplitter(Qt.Orientation.Horizontal); hsp.setHandleWidth(2)

        vw = QWidget(); vl = QVBoxLayout(vw)
        vl.setContentsMargins(4, 4, 2, 2); vl.setSpacing(0)
        self._vpS = Viewport("Open an image or drag & drop here")
        self._vpC = SwipeWidget(); self._vpC.hide()
        vl.addWidget(self._vpS); vl.addWidget(self._vpC)
        hsp.addWidget(vw)

        # right panel
        rp = QWidget(); rp.setFixedWidth(RIGHT_W)
        rpL = QVBoxLayout(rp); rpL.setContentsMargins(2, 4, 4, 4); rpL.setSpacing(5)

        # ── Fidelity card ──
        fc = self._card("Restoration Control")
        fl = fc.layout()

        fw_row = QHBoxLayout(); fw_row.setSpacing(3)
        fw_lbl = QLabel("Fidelity"); fw_lbl.setFixedWidth(SLD_LBL_W)
        fw_lbl.setStyleSheet("color:#999; font-size:10px;")
        self._sldFW = QSlider(Qt.Orientation.Horizontal)
        self._sldFW.setRange(0, 100); self._sldFW.setValue(50)
        self._lblFW = QLabel("50%"); self._lblFW.setFixedWidth(SLD_VAL_W)
        self._lblFW.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lblFW.setStyleSheet("color:#ff9999; font-weight:bold; font-size:10px;")
        self._sldFW.valueChanged.connect(lambda v: self._lblFW.setText("{}%".format(v)))
        fw_row.addWidget(fw_lbl); fw_row.addWidget(self._sldFW, stretch=1)
        fw_row.addWidget(self._lblFW)
        fl.addLayout(fw_row)

        hint = QLabel("0% = max AI restoration   100% = original")
        hint.setStyleSheet("color:#444; font-size:9px;")
        fl.addWidget(hint)
        rpL.addWidget(fc)

        # ── Metrics card ──
        mc = self._card("Quality Metrics")
        mg = QGridLayout(); mg.setSpacing(3)
        names = [("PSNR", True), ("SSIM", False),
                 ("Ent.B", False), ("Ent.A", False),
                 ("Col.B", False), ("Col.A", False)]
        self._mv = []
        for i, (nm, ac) in enumerate(names):
            fr, vl = _tile(nm, ac); mg.addWidget(fr, i // 2, i % 2)
            self._mv.append(vl)
        mc.layout().addLayout(mg); rpL.addWidget(mc)

        # ── Info card ──
        ic = self._card("Image Info")
        self._infoLbl = QLabel("—")
        self._infoLbl.setWordWrap(True)
        self._infoLbl.setStyleSheet("color:#666; font-size:10px;")
        ic.layout().addWidget(self._infoLbl)
        rpL.addWidget(ic)

        # ── Histogram ──
        hc = self._card("Histogram")
        self._histLbl = QLabel()
        self._histLbl.setFixedHeight(55)
        self._histLbl.setStyleSheet("background:#080808; border-radius:3px;")
        self._histLbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hc.layout().addWidget(self._histLbl); rpL.addWidget(hc)

        # ── Model status ──
        sc2 = self._card("Model Status")
        self._statusLbl = QLabel("GFPGANv1.3.pth — checking...")
        self._statusLbl.setWordWrap(True)
        self._statusLbl.setStyleSheet("color:#888; font-size:10px;")
        sc2.layout().addWidget(self._statusLbl); rpL.addWidget(sc2)
        self._check_model_status()

        rpL.addStretch()
        hsp.addWidget(rp)
        hsp.setStretchFactor(0, 1); hsp.setStretchFactor(1, 0)
        vsp.addWidget(hsp)

        # log
        lw = QWidget(); ll = QVBoxLayout(lw)
        ll.setContentsMargins(4, 2, 4, 4); ll.setSpacing(1)
        lt = QLabel("Processing Log (Explainability)")
        lt.setStyleSheet("color:#c53030; font-weight:bold; font-size:10px;")
        ll.addWidget(lt)
        self._logW = QTextEdit(); self._logW.setObjectName("log")
        self._logW.setReadOnly(True); ll.addWidget(self._logW)
        vsp.addWidget(lw)
        vsp.setStretchFactor(0, 5); vsp.setStretchFactor(1, 1)
        rl.addWidget(vsp, stretch=1)

        # ── Chat bar (Sprint 3 — Analysis §3.5.1 #3) ─────────────
        chat = QFrame(); chat.setObjectName("chat")
        chat.setStyleSheet(
            "QFrame#chat{background:#151515;border-top:1px solid #2a2a2a;"
            "min-height:34px;max-height:34px;}"
            "QLineEdit{background:#0f0f0f;color:#eee;border:1px solid #2a2a2a;"
            "border-radius:3px;padding:4px 8px;font-size:11px;}"
            "QLineEdit:focus{border-color:#c53030;}"
            "QLabel#chatprompt{color:#888;font-size:11px;padding:0 6px;}"
            "QPushButton#chatbtn{background:#c53030;color:#fff;border:none;"
            "border-radius:3px;padding:3px 14px;font-weight:bold;}"
            "QPushButton#chatbtn:hover{background:#dc4040;}")
        cl = QHBoxLayout(chat); cl.setContentsMargins(8, 3, 8, 3); cl.setSpacing(6)
        pmt = QLabel("▸ Chat"); pmt.setObjectName("chatprompt")
        cl.addWidget(pmt)
        self._chatIn = QLineEdit()
        self._chatIn.setPlaceholderText(
            "Try: 'restore', 'more ai', 'fidelity 40', 'compare', 'diff', "
            "'live', 'capture', 'batch', 'reset', 'help'")
        self._chatIn.returnPressed.connect(self._chat_send)
        cl.addWidget(self._chatIn, stretch=1)
        self._chatBtn = QPushButton("Send"); self._chatBtn.setObjectName("chatbtn")
        self._chatBtn.clicked.connect(self._chat_send)
        cl.addWidget(self._chatBtn)
        rl.addWidget(chat)

    # ── helpers ───────────────────────────────────────────────────

    def _card(self, title):
        f = QFrame(); f.setObjectName("card")
        l = QVBoxLayout(f)
        l.setContentsMargins(CARD_PAD, CARD_PAD - 2, CARD_PAD, CARD_PAD)
        l.setSpacing(4)
        t = QLabel(title)
        t.setStyleSheet("color:#c53030; font-weight:bold; font-size:10px;")
        l.addWidget(t)
        return f

    def _check_model_status(self):
        from face_restorer import DEFAULT_CHECKPOINT
        if os.path.isfile(DEFAULT_CHECKPOINT):
            sz = os.path.getsize(DEFAULT_CHECKPOINT) / 1e6
            self._statusLbl.setText(
                "GFPGANv1.3.pth: FOUND ({:.0f} MB) ✓".format(sz))
            self._statusLbl.setStyleSheet("color:#33cc66; font-size:10px;")
        else:
            self._statusLbl.setText(
                "GFPGANv1.3.pth: NOT FOUND\n"
                "Run: python scripts/download_model.py")
            self._statusLbl.setStyleSheet("color:#ff4444; font-size:10px;")

    @staticmethod
    def _pm(img):
        r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = r.shape
        return QPixmap.fromImage(
            QImage(r.data.tobytes(), w, h, c * w, QImage.Format.Format_RGB888))

    def _hist_pm(self):
        W, H = RIGHT_W - CARD_PAD * 2 - 4, 50
        c = np.zeros((H, W, 3), dtype=np.uint8); c[:] = (8, 8, 8)
        for img, cols in filter(lambda x: x[0] is not None, [
            (self._orig, [(80, 30, 100), (30, 100, 40), (40, 60, 160)]),
            (self._rest, [(190, 80, 150), (80, 190, 100), (100, 120, 220)]),
        ]):
            for ch, cl in enumerate(cols):
                h = cv2.calcHist([img], [ch], None, [256], [0, 256])
                cv2.normalize(h, h, 0, H - 3, cv2.NORM_MINMAX)
                pts = [(int(x * (W - 1) / 255), H - 1 - int(h[x][0]))
                       for x in range(256)]
                for j in range(1, len(pts)):
                    cv2.line(c, pts[j - 1], pts[j], cl, 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        return QPixmap.fromImage(
            QImage(rgb.data.tobytes(), W, H, 3 * W, QImage.Format.Format_RGB888))

    # ── view logic ────────────────────────────────────────────────

    def _chg_view(self, v):
        self._view = v
        self._vpS.setVisible(v != self.V_CMP)
        self._vpC.setVisible(v == self.V_CMP)
        self._show_view()

    def _show_view(self):
        v = self._view
        if v == self.V_ORIG:
            if self._orig is not None: self._vpS.img(self._pm(self._orig))
            else: self._vpS.clr(); self._vpS.setText("Open an image or drag & drop here")
        elif v == self.V_REST:
            if self._rest is not None: self._vpS.img(self._pm(self._rest))
            else: self._vpS.clr(); self._vpS.setText("Click 'Restore Face(s)' first")
        elif v == self.V_CMP:
            if self._orig is not None and self._rest is not None:
                self._vpC.set_images(self._pm(self._orig), self._pm(self._rest))
            else: self._vpC.clear()
        elif v == self.V_DIF:
            if self._dif is not None: self._vpS.img(self._pm(self._dif))
            else: self._vpS.clr(); self._vpS.setText("No difference map yet")

    def _upd_all(self):
        self._histLbl.setPixmap(self._hist_pm())
        self._show_view()

    def _set_met(self, m):
        if not m: return
        for lbl, v in zip(self._mv, [
            "{:.1f}".format(m.psnr), "{:.4f}".format(m.ssim),
            "{:.2f}".format(m.entropy_before), "{:.2f}".format(m.entropy_after),
            "{:.1f}".format(m.colorfulness_before), "{:.1f}".format(m.colorfulness_after),
        ]): lbl.setText(v)

    def _clr_met(self):
        for l in self._mv: l.setText("--")

    def _html_log(self, lines):
        o = "<pre style='font-family:Consolas;font-size:10px;line-height:1.35;'>"
        for ln in lines:
            e = ln.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            if "ERROR" in ln:      c = "#ff4444"
            elif "GFPGAN" in ln and "LOADED" in ln: c = "#33ff99"
            elif "Faces found" in ln or "Face " in ln: c = "#ffcc33"
            elif ln.strip().startswith("==="): c = "#c53030"
            elif ln.strip().startswith("---"): c = "#888"
            elif "[NOT ACTIVE]" in ln: c = "#555"
            elif "WARN" in ln:     c = "#ffaa33"
            else:                  c = "#aaa"
            o += "<span style='color:{}'>{}</span>\n".format(c, e)
        return o + "</pre>"

    # ── actions ───────────────────────────────────────────────────

    def _load(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "Images (*.jpg *.jpeg *.png *.bmp *.tiff);;All (*)")
        if p: self._open(p)

    def _open(self, path):
        im = cv2.imread(path)
        if im is None:
            self._logW.setHtml("<pre style='color:#f44'>[ERROR] Cannot read file</pre>")
            return
        self._orig = im; self._rest = self._dif = None
        self._view = self.V_ORIG; self._vbtns[0].setChecked(True); self._chg_view(0)
        self._bRest.setEnabled(True); self._bRset.setEnabled(True)
        self._bSave.setEnabled(False); self._clr_met()
        self._histLbl.setPixmap(QPixmap()); self._upd_all()

        h, w = im.shape[:2]
        pr = self._prof.profile(im)
        info = "{}×{}  {:.1f}MP".format(w, h, w * h / 1e6)
        if pr:
            info += "\nBrightness: {:.2f}  Contrast: {:.2f}".format(
                pr.brightness, pr.contrast)
            info += "\nBlur: {:.0f}  Noise: {:.1f}".format(pr.blur_score, pr.noise_level)
            info += "\nSkin: {:.0f}%  {}".format(pr.skin_ratio * 100,
                "• Face detected" if pr.has_skin else "")
            info += "\nFlags: {}".format(pr.summary() or "Normal")
        self._infoLbl.setText(info)

        ln = ["Loaded: {}  ({}×{}, {:.1f}MP)".format(
            os.path.basename(path), w, h, w * h / 1e6), ""]
        if pr:
            ln += ["--- analyzeImage() ---",
                "  brightness = {:.3f} {}".format(pr.brightness,
                    "(LOW)" if pr.is_low_light else "(HIGH)" if pr.is_overexposed else ""),
                "  contrast   = {:.3f}".format(pr.contrast),
                "  blur       = {:.1f} {}".format(pr.blur_score,
                    "(BLURRY)" if pr.is_blurry else "(SHARP)"),
                "  noise      = {:.1f} {}".format(pr.noise_level,
                    "(NOISY)" if pr.is_noisy else ""),
                "  skin/face  = {:.1f}% {}".format(pr.skin_ratio * 100,
                    "(DETECTED)" if pr.has_skin else ""),
                "", "Ready — click 'Restore Face(s)'"]
        self._logW.setHtml(self._html_log(ln))

    def _save(self):
        if self._rest is None: return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "restored.png",
                                            "PNG (*.png);;JPEG (*.jpg);;All (*)")
        if p: cv2.imwrite(p, self._rest)

    def _restore(self):
        if self._orig is None: return
        self.setCursor(Qt.CursorShape.WaitCursor)
        self._bRest.setEnabled(False); self._bRest.setText("Restoring...")
        QApplication.processEvents()

        fw = self._sldFW.value() / 100.0
        center_only = self._ckCenter.isChecked()

        self._worker = EnhanceWorker(self._pipe, self._orig, fw, center_only)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, result):
        if result.success and result.restored is not None:
            self._rest = result.restored
            self._dif  = result.diff_map
            self._bSave.setEnabled(True)
            self._view = self.V_CMP
            self._vbtns[2].setChecked(True); self._chg_view(self.V_CMP)
            self._upd_all(); self._set_met(result.metrics)
            if result.restoration:
                self._infoLbl.setText(
                    self._infoLbl.text() +
                    "\n\nFaces restored: {}".format(result.restoration.faces_found))

        self._logW.setHtml(self._html_log(result.log))
        self._logW.verticalScrollBar().setValue(
            self._logW.verticalScrollBar().maximum())
        self._bRest.setEnabled(True); self._bRest.setText("Restore Face(s)")
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _reset(self):
        self._rest = self._dif = None
        self._bSave.setEnabled(False); self._clr_met()
        self._histLbl.setPixmap(QPixmap()); self._logW.clear()
        self._view = self.V_ORIG; self._vbtns[0].setChecked(True); self._chg_view(0)

    # ── Batch mode (Sprint 2 — Analysis §3.5.1 #1, #2) ──────────

    def _batch(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select input folder — all images will be enhanced",
            os.path.expanduser("~"))
        if not folder:
            return

        files = collect_images(folder, recursive=False)
        if not files:
            self._logW.setHtml(self._html_log([
                "[BATCH] No supported images found in:",
                "  " + folder,
                "  (.jpg .jpeg .png .bmp .tiff .webp)",
            ]))
            return

        import time as _t
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        out_dir = os.path.join(
            proj_root, "outputs", "batch_" + _t.strftime("%Y%m%d_%H%M%S"))

        fidelity = self._sldFW.value() / 100.0
        only_center = self._ckCenter.isChecked()

        dlg = BatchDialog(files, out_dir, fidelity, only_center, parent=self)
        dlg.show()   # non-modal

        self._logW.setHtml(self._html_log([
            "=== Batch Enhancement Launched ===",
            "  Input:     {}".format(folder),
            "  Output:    {}".format(out_dir),
            "  Files:     {}".format(len(files)),
            "  Fidelity:  {:.2f}".format(fidelity),
            "  Center-only: {}".format(only_center),
            "",
            "Analysis Report §3.5.1 Scenarios #1 and #2",
        ]))

    # ── Chat (Sprint 3 — Analysis §3.5.1 #3) ────────────────────

    def _chat_send(self):
        text = self._chatIn.text().strip()
        if not text:
            return
        self._chatIn.clear()

        cmd = chat_parse(text)
        result = ChatExecutor.dispatch(cmd, self)

        # Append to the log as a small conversation thread.
        cur = self._logW.toHtml() or ""
        entry = (
            "<div style='color:#8ac'>&gt; {}</div>"
            "<div style='color:#bbb'>&nbsp;&nbsp;intent=<b>{}</b> "
            "&nbsp;{}</div>"
            "<div style='color:#7a7;margin-bottom:6px'>&nbsp;&nbsp;{}</div>"
        ).format(
            self._esc(text), cmd.intent,
            self._esc(str(cmd.params)) if cmd.params else "",
            self._esc(result))
        # If log currently holds non-chat content, keep last section small.
        if "chat-start" not in cur:
            cur = "<div id='chat-start' style='color:#555'>"\
                  "=== Chat Session ===</div>" + entry
        else:
            cur += entry
        self._logW.setHtml(cur)
        self._logW.verticalScrollBar().setValue(
            self._logW.verticalScrollBar().maximum())

    @staticmethod
    def _esc(s: str) -> str:
        return (str(s).replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;"))

    # ── Live camera (Analysis Report Scenario 4) ─────────────────

    def _toggle_live(self, checked: bool):
        if checked:
            self._start_live()
        else:
            self._stop_live()

    def _start_live(self):
        cams = CameraCapture.list_cameras(max_devices=3)
        if not cams:
            self._bLive.setChecked(False)
            self._logW.setHtml(self._html_log([
                "[ERROR] No camera device found.",
                "  Check that a webcam is connected and not used by another app.",
            ]))
            return

        if not self._cam.open(cams[0].index, width=1280, height=720):
            self._bLive.setChecked(False)
            self._logW.setHtml(self._html_log([
                "[ERROR] Failed to open camera at index {}".format(cams[0].index),
            ]))
            return

        # Switch to LIVE view
        self._view = self.V_LIVE
        self._vpS.setVisible(True)
        self._vpC.setVisible(False)
        for b in self._vbtns: b.setChecked(False)

        self._bShot.setVisible(True)
        self._filterLbl.setVisible(True)
        for b in self._fbtns: b.setVisible(True)
        self._bLoad.setEnabled(False); self._bRest.setEnabled(False)
        self._bRset.setEnabled(False)

        info = self._cam.info()
        self._infoLbl.setText("LIVE MODE\n" + info +
            "\n\nFilter: " + self._filter_name +
            "\nPress Capture to grab a frame\nand run GFPGAN enhancement.")

        self._logW.setHtml(self._html_log([
            "=== Live Camera Mode ===",
            "  {}".format(info),
            "  Available devices: {}".format(len(cams)),
            "  Preview running at ~30 FPS",
            "  Press 'Capture' to snapshot → GFPGAN pipeline",
        ]))
        self._cam_timer.start()

    def _stop_live(self):
        self._cam_timer.stop()
        self._cam.close()
        self._cam_frame = None

        self._bShot.setVisible(False)
        self._filterLbl.setVisible(False)
        for b in self._fbtns: b.setVisible(False)
        self._bLoad.setEnabled(True)
        self._bRset.setEnabled(self._orig is not None)
        self._bRest.setEnabled(self._orig is not None)

        self._vpC.setVisible(False)
        self._vpS.setVisible(True)
        self._view = self.V_ORIG
        self._vbtns[0].setChecked(True); self._chg_view(self.V_ORIG)

    def _cam_tick(self):
        frame = self._cam.read()
        if frame is None:
            return
        # Apply the active live filter before display (Sprint 4).
        display = frame
        try:
            display = self._filter.apply(frame)
            if display is None:
                display = frame
        except Exception:
            display = frame
        self._cam_frame = frame  # keep the raw frame for snapshot

        if self._filter_name == "ENHANCE":
            if not self._vpC.isVisible():
                self._vpS.setVisible(False)
                self._vpC.setVisible(True)
            self._vpC.set_images(self._pm(frame), self._pm(display))
        else:
            if not self._vpS.isVisible():
                self._vpS.setVisible(True)
                self._vpC.setVisible(False)
            self._vpS.img(self._pm(display))

    def _set_filter(self, idx: int):
        names = ["OFF", "BEAUTY", "ENHANCE", "AI"]
        if 0 <= idx < len(names):
            self._filter_name = names[idx]
            self._filter = make_filter(self._filter_name)
            if self._cam.is_open:
                self._infoLbl.setText(
                    "LIVE MODE\n{}\n\nFilter: {}\nPress Capture to grab a frame\n"
                    "and run GFPGAN enhancement.".format(
                        self._cam.info(), self._filter_name))

    def _capture_shot(self):
        if not self._cam.is_open:
            return
        snap = self._cam.snapshot()
        if snap is None:
            self._logW.setHtml(self._html_log(["[ERROR] Snapshot failed"]))
            return

        # If a live filter is active, the user probably wants its look
        # preserved in the snapshot. Apply it once to the stabilised frame.
        filter_used = self._filter_name
        if filter_used != "OFF":
            try:
                f = self._filter.apply(snap)
                if f is not None:
                    snap = f
            except Exception:
                pass

        self._cam_timer.stop()
        self._bLive.setChecked(False)
        self._stop_live()

        # Feed snapshot as if it were a loaded image
        self._orig = snap
        self._rest = self._dif = None
        self._bRest.setEnabled(True); self._bRset.setEnabled(True)
        self._bSave.setEnabled(False); self._clr_met()
        self._histLbl.setPixmap(QPixmap()); self._upd_all()

        h, w = snap.shape[:2]
        pr = self._prof.profile(snap)
        info = "Captured: {}x{}  {:.1f}MP".format(w, h, w * h / 1e6)
        if pr:
            info += "\nBrightness: {:.2f}  Contrast: {:.2f}".format(
                pr.brightness, pr.contrast)
            info += "\nBlur: {:.0f}  Noise: {:.1f}".format(pr.blur_score, pr.noise_level)
            info += "\nSkin: {:.0f}%  {}".format(pr.skin_ratio * 100,
                "• Face detected" if pr.has_skin else "")
            info += "\nFlags: {}".format(pr.summary() or "Normal")
        self._infoLbl.setText(info)

        self._logW.setHtml(self._html_log([
            "=== Snapshot Captured ===",
            "  Resolution: {}x{} ({:.1f} MP)".format(w, h, w * h / 1e6),
            "  Live filter: {}".format(filter_used),
            "",
            "Scenario 4: Real-Time Image Capture (Analysis §3.5.1)",
            "",
            "Ready — click 'Restore Face(s)' to run GFPGAN pipeline",
        ]))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        u = e.mimeData().urls()
        if u:
            p = u[0].toLocalFile()
            if p: self._open(p)
