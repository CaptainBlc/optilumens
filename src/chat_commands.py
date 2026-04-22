"""
Chat Commands — natural-language interface for OptiLumen GUI.

Implements Analysis Report §3.5.1 Scenario #3 ("AI chat prompt for photo
editing"). MVP is rule-based so it works offline, is fully explainable,
and is demo-safe. The same `Command` protocol later accepts an LLM
adapter — just swap `parse()` for an LLM-backed implementation.

Public API
----------
    cmd = parse(user_text)              # returns Command (never raises)
    cmd.intent                          # str: restore / set_fidelity / ...
    cmd.params                          # dict of typed arguments
    cmd.explanation                     # natural-language ack for user
    cmd.ok                              # True if intent was recognised

GUI contract
------------
    MainWindow binds each supported intent to one of its methods and
    calls `ChatExecutor.dispatch(cmd, window)`. Unknown commands return
    a `help` intent with a hint instead of crashing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

# ── Intent catalog ─────────────────────────────────────────────────

INTENTS = (
    "restore",         # run GFPGAN pipeline on loaded image
    "set_fidelity",    # params: pct 0..100
    "adjust_fidelity", # params: delta int (e.g. -20 for "more AI")
    "center_only",     # params: flag bool
    "reset",
    "save",
    "open",            # params: (prompts GUI file dialog)
    "live",            # toggle camera mode
    "capture",         # snapshot in live mode
    "batch",           # open batch dialog
    "view",            # params: which ∈ {original,restored,compare,diff}
    "help",
    "unknown",
)


@dataclass
class Command:
    intent:      str = "unknown"
    params:      Dict = field(default_factory=dict)
    explanation: str = ""
    raw_text:    str = ""

    @property
    def ok(self) -> bool:
        return self.intent != "unknown"


# ── Parser ─────────────────────────────────────────────────────────

# Helpers: every matcher returns a Command or None.

_FIDELITY_NUMBER_RE = re.compile(
    r"(?:fidelity|strength|etki|g(?:u|ü)c(?:u|ü)?)\D{0,10}(\d{1,3})", re.IGNORECASE)
_PERCENT_RE = re.compile(r"(\d{1,3})\s*%")

# Keyword buckets (lowercase). Turkish + English.
KW = {
    "restore":   ["restore", "enhance", "fix", "iyilestir", "iyileştir", "duzelt",
                  "düzelt", "onar", "gfpgan", "apply", "uygula", "yap", "çalıştır"],
    "reset":     ["reset", "sifirla", "sıfırla", "baştan", "basa don",
                  "revert", "undo"],
    "save":      ["save", "kaydet", "dışa aktar", "export"],
    "open":      ["open image", "dosya aç", "load", "resim yükle",
                  "fotograf ac", "fotoğraf aç"],
    "live":      ["live", "camera", "kamera", "webcam", "canlı"],
    "capture":   ["capture", "snapshot", "shot", "çek", "yakala", "cektir", "fotoğraf çek"],
    "batch":     ["batch", "folder", "klasor", "klasör", "toplu", "çoklu"],
    "cmp_orig":  ["original", "orijinal", "kaynak", "input"],
    "cmp_rest":  ["restored", "restore result", "enhanced", "iyilestirilmis",
                  "düzeltilmiş", "cikti"],
    "cmp_cmp":   ["compare", "karşılaştır", "karsilastir", "before after", "kıyas"],
    "cmp_dif":   ["diff", "difference", "fark", "heatmap", "ısı haritası"],
    "center_on": ["only center", "center face", "merkez yuz", "merkez yüz",
                  "just center"],
    "center_off":["all faces", "tüm yüzler", "tum yuzler", "every face"],
    "more_ai":   ["more ai", "stronger", "daha ai", "daha yapay", "daha güçlü",
                  "daha guclu", "agresif", "aggressive", "more enhance",
                  "less fidelity"],
    "less_ai":   ["less ai", "softer", "daha az ai", "daha doğal", "daha dogal",
                  "natural", "hafif", "lighter", "more fidelity", "subtle"],
    "help":      ["help", "yardım", "yardim", "what can", "ne yapabilir",
                  "?", "commands"],
}


def _any(text: str, words) -> bool:
    return any(w in text for w in words)


def _normalise(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("ı", "i")  # collapse Turkish 'ı' → 'i' for matching
    return s


def parse(user_text: str) -> Command:
    """Convert free-form user text into a typed Command (never raises)."""
    if not user_text or not user_text.strip():
        return Command(intent="unknown", raw_text=user_text or "",
                       explanation="Empty command.")

    raw  = user_text
    text = _normalise(user_text)

    # 1) help — always match first
    if _any(text, KW["help"]):
        return Command(
            intent="help", raw_text=raw,
            explanation=(
                "Try: 'restore', 'more ai', 'less ai', 'fidelity 40', "
                "'compare', 'diff', 'reset', 'save', 'live', 'capture', "
                "'batch', 'only center face'."
            ))

    # 2) explicit fidelity number
    m = _FIDELITY_NUMBER_RE.search(text) or _PERCENT_RE.search(text)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return Command(
            intent="set_fidelity", params={"pct": pct}, raw_text=raw,
            explanation="Fidelity set to {}%.".format(pct))

    # 3) relative fidelity adjustments
    if _any(text, KW["more_ai"]):
        return Command(
            intent="adjust_fidelity", params={"delta": -20}, raw_text=raw,
            explanation="Pushing fidelity down by 20% (more AI restoration).")
    if _any(text, KW["less_ai"]):
        return Command(
            intent="adjust_fidelity", params={"delta": +20}, raw_text=raw,
            explanation="Pulling fidelity up by 20% (closer to original).")

    # 4) view switch
    if _any(text, KW["cmp_cmp"]):
        return Command(intent="view", params={"which": "compare"},
                       raw_text=raw, explanation="Showing Compare view.")
    if _any(text, KW["cmp_dif"]):
        return Command(intent="view", params={"which": "diff"},
                       raw_text=raw, explanation="Showing Difference map.")
    if _any(text, KW["cmp_orig"]):
        return Command(intent="view", params={"which": "original"},
                       raw_text=raw, explanation="Showing Original.")
    if _any(text, KW["cmp_rest"]):
        return Command(intent="view", params={"which": "restored"},
                       raw_text=raw, explanation="Showing Restored.")

    # 5) center-only toggle
    if _any(text, KW["center_on"]):
        return Command(intent="center_only", params={"flag": True},
                       raw_text=raw, explanation="Only center face will be restored.")
    if _any(text, KW["center_off"]):
        return Command(intent="center_only", params={"flag": False},
                       raw_text=raw, explanation="All detected faces will be restored.")

    # 6) actions (order matters: capture before live in case of 'capture live')
    if _any(text, KW["capture"]):
        return Command(intent="capture", raw_text=raw,
                       explanation="Capturing a snapshot.")
    if _any(text, KW["live"]):
        return Command(intent="live", raw_text=raw,
                       explanation="Toggling live camera mode.")
    if _any(text, KW["batch"]):
        return Command(intent="batch", raw_text=raw,
                       explanation="Opening batch folder picker.")
    if _any(text, KW["save"]):
        return Command(intent="save", raw_text=raw,
                       explanation="Saving the current restored image.")
    if _any(text, KW["open"]):
        return Command(intent="open", raw_text=raw,
                       explanation="Opening file picker.")
    if _any(text, KW["reset"]):
        return Command(intent="reset", raw_text=raw,
                       explanation="Reverting to the original image.")
    if _any(text, KW["restore"]):
        return Command(intent="restore", raw_text=raw,
                       explanation="Running GFPGAN restoration.")

    # 7) fallback
    return Command(
        intent="unknown", raw_text=raw,
        explanation="Sorry, I did not understand. Type 'help' for commands.")


# ── Executor (GUI glue) ────────────────────────────────────────────

class ChatExecutor:
    """
    Calls MainWindow methods for a parsed Command.

    Kept separate from the parser so the parser is purely functional
    and unit-testable without Qt.
    """

    VIEW_MAP = {
        "original": 0, "restored": 1, "compare": 2, "diff": 3,
    }

    @staticmethod
    def dispatch(cmd: Command, w) -> str:
        """
        Execute `cmd` against MainWindow `w`. Returns a user-facing
        status string. Never raises — always reports the outcome.
        """
        try:
            i = cmd.intent
            p = cmd.params

            if i == "set_fidelity":
                pct = int(p.get("pct", 50))
                w._sldFW.setValue(pct)
                return "Fidelity = {}%.".format(pct)

            if i == "adjust_fidelity":
                cur = w._sldFW.value()
                new = max(0, min(100, cur + int(p.get("delta", 0))))
                w._sldFW.setValue(new)
                return "Fidelity {} → {}%.".format(cur, new)

            if i == "center_only":
                flag = bool(p.get("flag", False))
                w._ckCenter.setChecked(flag)
                return "Center-only: {}.".format("ON" if flag else "OFF")

            if i == "view":
                which = str(p.get("which", "original"))
                idx = ChatExecutor.VIEW_MAP.get(which, 0)
                if idx < len(w._vbtns):
                    w._vbtns[idx].setChecked(True)
                    w._chg_view(idx)
                return "View: {}.".format(which)

            if i == "restore":
                if w._orig is None:
                    return "No image loaded — try 'open' or 'live' first."
                w._restore()
                return "Restoration started."

            if i == "reset":
                w._reset()
                return "Reset done."

            if i == "save":
                if w._rest is None:
                    return "Nothing to save — run 'restore' first."
                w._save()
                return "Save dialog opened."

            if i == "open":
                w._load()
                return "Open dialog shown."

            if i == "live":
                w._bLive.setChecked(not w._bLive.isChecked())
                w._toggle_live(w._bLive.isChecked())
                return "Live mode: {}.".format(
                    "ON" if w._bLive.isChecked() else "OFF")

            if i == "capture":
                if not w._cam.is_open:
                    return "Camera not active — say 'live' first."
                w._capture_shot()
                return "Snapshot captured."

            if i == "batch":
                w._batch()
                return "Batch picker opened."

            if i == "help":
                return cmd.explanation

            return cmd.explanation or "Unknown command."

        except Exception as ex:
            return "Command failed: {}".format(ex)
