"""
PyQt6 GUI for the Pixel Enhancement Module.
CMPE 491 Senior Design Project — Hybrid Image Enhancement System.
"""

import os
import sys

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QCheckBox, QTextEdit, QFrame,
    QFileDialog, QSplitter, QApplication, QSizePolicy,
    QGridLayout, QButtonGroup,
)
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import (
    QPixmap, QImage, QAction, QKeySequence, QPainter,
    QPen, QColor, QFont, QBrush, QRegion,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pixel_pipeline import ImageProcessor
from profiler import ImageProfiler

# ── Sizes ──────────────────────────────────────────────────────────
RIGHT_W = 260          # fixed right-panel width
SLD_LABEL_W = 58       # slider name column
SLD_VAL_W = 38         # slider value column ("100%")
CARD_PAD = 8           # card inner padding
CARD_GAP = 4           # gap between cards
TB_H = 36              # toolbar height

STYLE = """
* { font-family: "Segoe UI", Arial; font-size: 11px; }
QMainWindow { background: #0f0e17; }

QMenuBar { background: #13122a; color: #a7a9be; }
QMenuBar::item:selected { background: #2e2b4a; }
QMenu { background: #1a1932; border: 1px solid #2e2b4a; color: #ddd; }
QMenu::item:selected { background: #7c3aed; }

QFrame#toolbar {
    background: #13122a; border-bottom: 2px solid #7c3aed;
    min-height: """ + str(TB_H) + """px; max-height: """ + str(TB_H) + """px;
}

QPushButton {
    background: #2e2b4a; color: #e0def4; border: 1px solid #3d3960;
    border-radius: 4px; padding: 4px 10px;
}
QPushButton:hover { background: #3d3960; border-color: #7c3aed; }
QPushButton:pressed { background: #7c3aed; }
QPushButton:disabled { background: #1a1932; color: #444; border-color: #252340; }
QPushButton#accent {
    background: #7c3aed; color: #fff; border: none;
    font-weight: bold; padding: 4px 16px; border-radius: 5px;
}
QPushButton#accent:hover { background: #8b5cf6; }
QPushButton#accent:disabled { background: #2e2b4a; color: #555; }
QPushButton#vb {
    background: none; color: #6e6a86; border: 1px solid #2e2b4a;
    border-radius: 3px; padding: 3px 8px; font-size: 10px;
}
QPushButton#vb:checked { background: #7c3aed; color: #fff; border-color: #7c3aed; }
QPushButton#vb:hover { border-color: #7c3aed; color: #c4b5fd; }

QFrame#card {
    background: #161530; border: 1px solid #2e2b4a; border-radius: 6px;
}
QSlider::groove:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #2e2b4a,stop:1 #7c3aed);
    height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #e0def4; width: 12px; height: 12px;
    margin: -4px 0; border-radius: 6px; border: 2px solid #7c3aed;
}
QSlider::handle:horizontal:disabled { background: #444; border-color: #333; }

QCheckBox { color: #a7a9be; spacing: 4px; }
QCheckBox::indicator {
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid #3d3960; background: #1a1932;
}
QCheckBox::indicator:checked { background: #7c3aed; border-color: #7c3aed; }

QTextEdit#log {
    background: #0a0914; color: #c4b5fd;
    font-family: Consolas, monospace; font-size: 10px;
    border: 1px solid #2e2b4a; border-radius: 4px; padding: 3px;
}
QScrollBar:vertical { background: #0f0e17; width: 6px; }
QScrollBar::handle:vertical { background: #3d3960; min-height: 20px; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QLabel { color: #e0def4; }
"""


# ── Swipe Compare Widget ──────────────────────────────────────────

class SwipeWidget(QWidget):
    """Before/After comparison with draggable divider."""

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

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor("#12111f"))

        if not self._bef or not self._aft:
            p.setPen(QColor("#6e6a86"))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Enhance first, then Compare")
            p.end()
            return

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

        # divider
        p.setPen(QPen(QColor("#7c3aed"), 2))
        p.drawLine(lx, yo, lx, yo + ih)

        # handle
        cy = yo + ih // 2
        p.setBrush(QBrush(QColor("#7c3aed")))
        p.setPen(QPen(QColor("#fff"), 1))
        p.drawEllipse(QPoint(lx, cy), 12, 12)
        p.setPen(QPen(QColor("#fff"), 2))
        for dx, d in [(-5, -1), (5, 1)]:
            p.drawLine(lx + dx, cy, lx + dx + d * 3, cy - 3)
            p.drawLine(lx + dx, cy, lx + dx + d * 3, cy + 3)

        # labels
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.setPen(QColor("#ff8c00"))
        p.drawText(xo + 6, yo + 15, "BEFORE")
        p.setPen(QColor("#34d399"))
        p.drawText(xo + iw - 48, yo + 15, "AFTER")

        # badge
        pct = int(self._ratio * 100)
        p.setFont(QFont("Segoe UI", 8))
        p.setBrush(QBrush(QColor(0, 0, 0, 170)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(lx - 16, yo + ih - 20, 32, 14, 3, 3)
        p.setPen(QColor("#fff"))
        p.drawText(QRect(lx - 16, yo + ih - 20, 32, 14),
                   Qt.AlignmentFlag.AlignCenter, "{}%".format(pct))
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._ir.contains(e.pos()):
            self._drag = True
            self._upd(e.pos())

    def mouseMoveEvent(self, e):
        self.setCursor(Qt.CursorShape.SplitHCursor
                       if self._ir.contains(e.pos())
                       else Qt.CursorShape.ArrowCursor)
        if self._drag:
            self._upd(e.pos())

    def mouseReleaseEvent(self, _):
        self._drag = False

    def _upd(self, pos):
        if self._ir.width() < 1:
            return
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
        self.setStyleSheet("background:#12111f; border:1px solid #2e2b4a;"
                           "border-radius:6px; color:#6e6a86;")

    def img(self, pm):
        self._px = pm
        self._fit()

    def clr(self):
        self._px = None
        super().setPixmap(QPixmap())

    def _fit(self):
        if self._px and not self._px.isNull():
            super().setPixmap(self._px.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()


# ── Metric tile ───────────────────────────────────────────────────

def _tile(label, accent=False):
    f = QFrame()
    f.setFixedHeight(44)
    f.setStyleSheet("background:{}; border-radius:4px;".format(
        "#2d1a6b" if accent else "#1c1a35"))
    l = QVBoxLayout(f)
    l.setContentsMargins(4, 2, 4, 2)
    l.setSpacing(0)
    v = QLabel("--")
    v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    v.setStyleSheet("font-size:13px; font-weight:bold; color:#e0def4;")
    n = QLabel(label)
    n.setAlignment(Qt.AlignmentFlag.AlignCenter)
    n.setStyleSheet("font-size:9px; color:#6e6a86;")
    l.addWidget(v)
    l.addWidget(n)
    return f, v


# ── MainWindow ────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    V_ORIG = 0
    V_ENH = 1
    V_CMP = 2
    V_DIF = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Enhancement System")
        self.setMinimumSize(960, 620)
        self.resize(1280, 760)
        self.setAcceptDrops(True)

        self._orig = self._enh = self._dif = None
        self._proc = ImageProcessor()
        self._prof = ImageProfiler()
        self._view = self.V_ORIG

        self._build_menu()
        self._build_ui()
        self.setStyleSheet(STYLE)

    def _build_menu(self):
        fm = self.menuBar().addMenu("File")
        for t, sc, fn in [("Open", "Ctrl+O", self._load),
                           ("Save", "Ctrl+S", self._save),
                           ("sep", "", None),
                           ("Quit", "Ctrl+Q", self.close)]:
            if t == "sep":
                fm.addSeparator()
            else:
                a = QAction(t, self)
                a.setShortcut(QKeySequence(sc))
                a.triggered.connect(fn)
                fm.addAction(a)

    # ── build ─────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # ──── toolbar ────
        tb = QFrame()
        tb.setObjectName("toolbar")
        tl = QHBoxLayout(tb)
        tl.setContentsMargins(6, 3, 6, 3)
        tl.setSpacing(4)

        self._bLoad = QPushButton("Open")
        self._bLoad.clicked.connect(self._load)
        self._bSave = QPushButton("Save")
        self._bSave.setEnabled(False)
        self._bSave.clicked.connect(self._save)
        self._bEnh = QPushButton("Enhance")
        self._bEnh.setObjectName("accent")
        self._bEnh.setEnabled(False)
        self._bEnh.clicked.connect(self._enhance)
        self._bRst = QPushButton("Reset")
        self._bRst.setEnabled(False)
        self._bRst.clicked.connect(self._reset)

        for b in (self._bLoad, self._bSave, self._bEnh, self._bRst):
            tl.addWidget(b)

        tl.addSpacing(8)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#2e2b4a;")
        tl.addWidget(sep)
        tl.addSpacing(4)

        self._vbtns = []
        self._vgrp = QButtonGroup(self)
        self._vgrp.setExclusive(True)
        for i, nm in enumerate(["Original", "Enhanced", "Compare", "Diff"]):
            b = QPushButton(nm)
            b.setObjectName("vb")
            b.setCheckable(True)
            if i == 0:
                b.setChecked(True)
            self._vgrp.addButton(b, i)
            self._vbtns.append(b)
            tl.addWidget(b)
        self._vgrp.idClicked.connect(self._chg_view)

        tl.addStretch()

        self._ckAuto = QCheckBox("Auto")
        self._ckAuto.setChecked(True)
        self._ckAuto.stateChanged.connect(self._tog_auto)
        self._ckLive = QCheckBox("Live")
        tl.addWidget(self._ckAuto)
        tl.addWidget(self._ckLive)

        rl.addWidget(tb)

        # ──── body ────
        vsp = QSplitter(Qt.Orientation.Vertical)
        vsp.setHandleWidth(2)

        hsp = QSplitter(Qt.Orientation.Horizontal)
        hsp.setHandleWidth(2)

        # viewport
        vw = QWidget()
        vl = QVBoxLayout(vw)
        vl.setContentsMargins(4, 4, 2, 2)
        vl.setSpacing(0)
        self._vpSimple = Viewport("Open an image or drag & drop")
        self._vpSwipe = SwipeWidget()
        self._vpSwipe.hide()
        vl.addWidget(self._vpSimple)
        vl.addWidget(self._vpSwipe)
        hsp.addWidget(vw)

        # ──── right panel ────
        rp = QWidget()
        rp.setFixedWidth(RIGHT_W)
        rpL = QVBoxLayout(rp)
        rpL.setContentsMargins(2, 4, 4, 4)
        rpL.setSpacing(CARD_GAP)

        # controls
        c1 = self._mk_card("Controls")
        c1l = c1.layout()
        self._sS, _ = self._mk_sld(c1l, "Sharp", 50)
        self._sN, _ = self._mk_sld(c1l, "Denoise", 0)
        self._sD, _ = self._mk_sld(c1l, "Detail", 50)
        self._sB, _ = self._mk_sld(c1l, "Smooth", 50)
        self._sliders_on(False)
        rpL.addWidget(c1)

        # metrics
        c2 = self._mk_card("Quality")
        g = QGridLayout()
        g.setSpacing(3)
        names = [("PSNR", True), ("SSIM", False),
                 ("Ent.B", False), ("Ent.A", False),
                 ("Col.B", False), ("Col.A", False)]
        self._mvals = []
        for i, (nm, ac) in enumerate(names):
            fr, vl = _tile(nm, ac)
            g.addWidget(fr, i // 2, i % 2)
            self._mvals.append(vl)
        c2.layout().addLayout(g)
        rpL.addWidget(c2)

        # histogram
        c3 = self._mk_card("Histogram")
        self._hLbl = QLabel("B=solid  A=bright")
        self._hLbl.setFixedHeight(55)
        self._hLbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hLbl.setStyleSheet(
            "background:#0a0914; border-radius:3px; color:#6e6a86; font-size:9px;")
        c3.layout().addWidget(self._hLbl)
        rpL.addWidget(c3)

        rpL.addStretch()
        hsp.addWidget(rp)

        hsp.setStretchFactor(0, 1)
        hsp.setStretchFactor(1, 0)
        vsp.addWidget(hsp)

        # log
        lw = QWidget()
        lwl = QVBoxLayout(lw)
        lwl.setContentsMargins(4, 2, 4, 4)
        lwl.setSpacing(1)
        lt = QLabel("Processing Log")
        lt.setStyleSheet("color:#7c3aed; font-weight:bold; font-size:10px;")
        lwl.addWidget(lt)
        self._logW = QTextEdit()
        self._logW.setObjectName("log")
        self._logW.setReadOnly(True)
        lwl.addWidget(self._logW)
        vsp.addWidget(lw)

        vsp.setStretchFactor(0, 5)
        vsp.setStretchFactor(1, 1)
        rl.addWidget(vsp, stretch=1)

    # ── helpers ───────────────────────────────────────────────────

    def _mk_card(self, title):
        f = QFrame()
        f.setObjectName("card")
        l = QVBoxLayout(f)
        l.setContentsMargins(CARD_PAD, CARD_PAD - 2, CARD_PAD, CARD_PAD)
        l.setSpacing(CARD_GAP)
        t = QLabel(title)
        t.setStyleSheet("color:#7c3aed; font-weight:bold; font-size:10px;")
        l.addWidget(t)
        return f

    def _mk_sld(self, parent, name, default):
        row = QHBoxLayout()
        row.setSpacing(3)
        lb = QLabel(name)
        lb.setFixedWidth(SLD_LABEL_W)
        lb.setStyleSheet("color:#a7a9be; font-size:10px;")
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(0, 100)
        s.setValue(default)
        vl = QLabel("{}%".format(default))
        vl.setFixedWidth(SLD_VAL_W)
        vl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        vl.setStyleSheet("color:#c4b5fd; font-weight:bold; font-size:10px;")
        s.valueChanged.connect(lambda v, lb=vl: lb.setText("{}%".format(v)))
        s.valueChanged.connect(self._sld_chg)
        row.addWidget(lb)
        row.addWidget(s, stretch=1)
        row.addWidget(vl)
        parent.addLayout(row)
        return s, vl

    def _sliders_on(self, on):
        for s in (self._sS, self._sN, self._sD, self._sB):
            s.setEnabled(on)

    # ── conversions ───────────────────────────────────────────────

    @staticmethod
    def _pm(img):
        r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = r.shape
        return QPixmap.fromImage(QImage(r.data.tobytes(), w, h, c * w,
                                         QImage.Format.Format_RGB888))

    def _hist_overlay(self):
        W, H = RIGHT_W - CARD_PAD * 2 - 4, 50
        c = np.zeros((H, W, 3), dtype=np.uint8)
        c[:] = (10, 9, 20)
        sets = []
        if self._orig is not None:
            sets.append((self._orig, [(140, 40, 100), (40, 140, 50), (60, 70, 180)]))
        if self._enh is not None:
            sets.append((self._enh, [(220, 100, 170), (100, 220, 120), (120, 140, 255)]))
        for img, cols in sets:
            for ch, cl in enumerate(cols):
                h = cv2.calcHist([img], [ch], None, [256], [0, 256])
                cv2.normalize(h, h, 0, H - 3, cv2.NORM_MINMAX)
                pts = [(int(x * (W - 1) / 255), H - 1 - int(h[x][0]))
                       for x in range(256)]
                for j in range(1, len(pts)):
                    cv2.line(c, pts[j - 1], pts[j], cl, 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        return QPixmap.fromImage(QImage(rgb.data.tobytes(), W, H, 3 * W,
                                         QImage.Format.Format_RGB888))

    # ── view logic ────────────────────────────────────────────────

    def _chg_view(self, v):
        self._view = v
        sw = v == self.V_CMP
        self._vpSimple.setVisible(not sw)
        self._vpSwipe.setVisible(sw)
        self._show_view()

    def _show_view(self):
        v = self._view
        if v == self.V_ORIG:
            if self._orig is not None:
                self._vpSimple.img(self._pm(self._orig))
            else:
                self._vpSimple.clr()
                self._vpSimple.setText("Open an image or drag & drop")
        elif v == self.V_ENH:
            if self._enh is not None:
                self._vpSimple.img(self._pm(self._enh))
            else:
                self._vpSimple.clr()
                self._vpSimple.setText("Click Enhance first")
        elif v == self.V_CMP:
            if self._orig is not None and self._enh is not None:
                self._vpSwipe.set_images(self._pm(self._orig), self._pm(self._enh))
            else:
                self._vpSwipe.clear()
        elif v == self.V_DIF:
            if self._dif is not None:
                self._vpSimple.img(self._pm(self._dif))
            else:
                self._vpSimple.clr()
                self._vpSimple.setText("No difference map yet")

    def _refresh_all(self):
        self._hLbl.setPixmap(self._hist_overlay())
        self._show_view()

    def _set_met(self, m):
        if not m:
            return
        for lbl, v in zip(self._mvals, [
            "{:.1f}".format(m.psnr), "{:.4f}".format(m.ssim),
            "{:.2f}".format(m.entropy_original), "{:.2f}".format(m.entropy_enhanced),
            "{:.1f}".format(m.colorfulness_original), "{:.1f}".format(m.colorfulness_enhanced),
        ]):
            lbl.setText(v)

    def _clr_met(self):
        for lbl in self._mvals:
            lbl.setText("--")

    def _html_log(self, lines):
        o = "<pre style='font-family:Consolas;font-size:10px;line-height:1.35;'>"
        for ln in lines:
            e = ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if "ERROR" in ln:
                c = "#f43f5e"
            elif "APPLIED" in ln:
                c = "#34d399"
            elif "SKIPPED" in ln:
                c = "#fbbf24"
            elif ln.strip().startswith("---"):
                c = "#818cf8"
            elif "MANUAL" in ln or "Mode:" in ln:
                c = "#c084fc"
            elif ln.strip().startswith("["):
                c = "#7dd3fc"
            else:
                c = "#a7a9be"
            o += "<span style='color:{}'>{}</span>\n".format(c, e)
        return o + "</pre>"

    # ── actions ───────────────────────────────────────────────────

    def _load(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff);;All (*)")
        if p:
            self._open(p)

    def _open(self, path):
        im = cv2.imread(path)
        if im is None:
            self._logW.setHtml("<pre style='color:#f43f5e'>[ERROR] Cannot read</pre>")
            return
        self._orig = im
        self._enh = self._dif = None
        self._view = self.V_ORIG
        self._vbtns[0].setChecked(True)
        self._chg_view(0)
        self._bEnh.setEnabled(True)
        self._bRst.setEnabled(True)
        self._bSave.setEnabled(False)
        self._clr_met()
        self._refresh_all()

        h, w = im.shape[:2]
        pr = self._prof.profile(im)
        ln = ["Loaded: {} ({}x{}, {:.1f}MP)".format(
            os.path.basename(path), w, h, w * h / 1e6), ""]
        if pr:
            ln += [
                "--- Input Analysis ---",
                "  brightness = {:.3f} {}".format(pr.brightness,
                    "(LOW)" if pr.is_low_light else "(HIGH)" if pr.is_overexposed else ""),
                "  contrast   = {:.3f}".format(pr.contrast),
                "  blur       = {:.1f} {}".format(pr.blur_score,
                    "(BLURRY)" if pr.is_blurry else "(SHARP)"),
                "  noise      = {:.1f} {}".format(pr.noise_level,
                    "(NOISY)" if pr.is_noisy else ""),
                "  skin       = {:.3f} {}".format(pr.skin_ratio,
                    "(SKIN)" if pr.has_skin else ""),
                "", "Ready.",
            ]
        self._logW.setHtml(self._html_log(ln))

    def _save(self):
        if self._enh is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "enhanced.png",
                                            "PNG (*.png);;JPEG (*.jpg);;All (*)")
        if p:
            cv2.imwrite(p, self._enh)

    def _enhance(self):
        if self._orig is None:
            return
        self.setCursor(Qt.CursorShape.WaitCursor)
        self._bEnh.setEnabled(False)
        QApplication.processEvents()

        md = None
        if not self._ckAuto.isChecked():
            s, n, d, b = (self._sS.value(), self._sN.value(),
                          self._sD.value(), self._sB.value())
            md = {
                "denoise": n > 5,
                "denoise_h": n / 100.0 * 20.0,
                "denoise_roi_weight": 0.4,
                "sharpen_strength": s / 100.0 * 3.0,
                "sharpen_sigma": 1.5,
                "clahe_clip": d / 100.0 * 4.0,
                "bg_blend": b / 100.0,
            }

        r = self._proc.enhanceImage(
            self._orig, mask_map=None, auto_mask=True, manual_decisions=md)

        if r.success and r.enhanced is not None:
            self._enh = r.enhanced
            self._dif = r.difference_map
            self._bSave.setEnabled(True)
            self._view = self.V_CMP
            self._vbtns[2].setChecked(True)
            self._chg_view(self.V_CMP)
            self._refresh_all()
            self._set_met(r.metrics)

        self._logW.setHtml(self._html_log(r.log))
        self._logW.verticalScrollBar().setValue(
            self._logW.verticalScrollBar().maximum())
        self._bEnh.setEnabled(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _reset(self):
        self._enh = self._dif = None
        self._bSave.setEnabled(False)
        self._clr_met()
        self._hLbl.setPixmap(QPixmap())
        self._hLbl.setText("B=solid  A=bright")
        self._logW.clear()
        self._view = self.V_ORIG
        self._vbtns[0].setChecked(True)
        self._chg_view(0)

    def _tog_auto(self, _):
        self._sliders_on(not self._ckAuto.isChecked())

    def _sld_chg(self, _):
        if (self._ckLive.isChecked() and not self._ckAuto.isChecked()
                and self._orig is not None):
            self._enhance()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        u = e.mimeData().urls()
        if u:
            p = u[0].toLocalFile()
            if p:
                self._open(p)
