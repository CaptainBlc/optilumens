"""
Chat Commands — natural-language interface for OptiLumen.

Goes well beyond the keyword-matcher MVP: this parser understands free,
conversational English/Turkish requests and turns them into a typed,
explainable pipeline configuration. No external API or LLM is needed —
the parser is fully local, deterministic, and offline-ready.

Capabilities
------------
1. **Compound sentences** — "smooth her skin a lot and sharpen the eyes"
   yields *two* intents in one Command.
2. **Magnitude detection** — "a little", "a bit", "more", "a lot",
   "very", "extremely" map to a 0..1 score.
3. **Region targeting** — face, skin, eyes, lips, brows, hair,
   background, all.
4. **Scene presets** — "fix this old photo", "make it look professional",
   "magazine cover", "natural", "vivid" map to fully-specified
   `ScenePreset` configurations.
5. **Conversational acks** — replies feel friendly, not robotic.
6. **Help & status** — "what can you do?", "how does it work?",
   "explain the result".

Public API (back-compatible with the old executor)
--------------------------------------------------
    cmd = parse(user_text)              # returns Command (never raises)
    cmd.intent                          # str
    cmd.params                          # dict (typed)
    cmd.intents                         # list[Command] for compound input
    cmd.explanation                     # natural-language ack
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from scene_presets import ScenePreset, find_preset_by_trigger, list_presets


# ── Intent catalog ────────────────────────────────────────────────────

INTENTS = (
    "restore",          # run pipeline
    "set_fidelity",     # params: pct 0..100
    "adjust_fidelity",  # params: delta int
    "preset",           # params: name str (one of scene_presets)
    "tweak_region",     # params: region, action, magnitude
    "view",             # params: which ∈ original/restored/compare/diff
    "center_only",      # params: flag bool
    "set_filter",       # params: name ∈ OFF/BEAUTY/ENHANCE/AI (live mode)
    "save", "open", "reset", "live", "capture", "batch", "help",
    "status",           # explain current image / last result
    "greet",            # "hi", "merhaba"
    "thanks",           # ack
    "compound",         # carries .intents = [Command, ...]
    "unknown",
)


@dataclass
class Command:
    intent:      str = "unknown"
    params:      Dict = field(default_factory=dict)
    explanation: str = ""
    raw_text:    str = ""
    intents:     List["Command"] = field(default_factory=list)  # for compound

    @property
    def ok(self) -> bool:
        return self.intent != "unknown"


# ── Lexicon: intents, regions, magnitudes, actions ───────────────────

_NORMAL_TR = str.maketrans({
    "ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c",
    "İ": "i", "Ş": "s", "Ğ": "g", "Ü": "u", "Ö": "o", "Ç": "c",
})

def _normalise(s: str) -> str:
    return s.strip().lower().translate(_NORMAL_TR)


# Magnitude phrases → 0..1 score
MAGNITUDE = [
    # very strong
    (("extremely", "very strong", "very much", "way more", "a ton",
      "super", "max", "maximum", "full", "asiri", "cok fazla", "tamamen",
      "tam", "full"), 0.95),
    # strong
    (("a lot", "much more", "lots", "very", "really", "heavily",
      "strong", "strongly", "aggressive", "aggressively", "epey", "fazla",
      "iyice", "cok"), 0.80),
    # medium-high
    (("more", "harder", "stronger", "increase", "boost", "push",
      "biraz daha", "daha fazla", "artir", "yukselt"), 0.65),
    # medium
    (("medium", "balanced", "moderate", "ortala", "ortalama"), 0.50),
    # mild
    (("a bit", "a little", "slightly", "softly", "soft", "gently",
      "gentle", "subtle", "lightly", "light", "small", "minor", "mild",
      "barely", "biraz", "az", "yumusak", "hafif"), 0.30),
    # very mild
    (("hardly", "almost none", "barely any", "very little", "minimal",
      "neredeyse hic", "cok az", "minimal"), 0.15),
]


def _magnitude(text: str) -> Optional[float]:
    """Find the strongest magnitude phrase in text. None if absent."""
    for phrases, score in MAGNITUDE:
        for p in phrases:
            if p in text:
                return score
    return None


# Region terms → canonical region key
REGION_KEYWORDS = {
    # Note: keywords are matched as prefixes (\b<word>\w*) so Turkish
    # agglutinative suffixes work — e.g. "cild" matches "cildini",
    # "goz" matches "gozler" / "gozleri", "yuz" matches "yuzu" /
    # "yuzunu". Keep the shortest stable stem here.
    "skin":       ["skin", "complexion", "cilt", "cild", "ten ", "teni"],
    "eyes":       ["eye", "iris", "pupil", "goz"],
    "lips":       ["lip", "mouth", "dudak", "agiz", "ağız"],
    "brows":      ["brow", "eyebrow", "kas ", "kasl"],
    "nose":       ["nose", "burun", "burnu"],
    "hair":       ["hair", "sac ", "saç ", "saçl", "sacl"],
    "face":       ["face", "yuz", "cehre", "surat"],
    "background": ["background", "behind", "scenery", "arka plan", "arkaplan",
                   "arka taraf"],
    "all":        ["everything", "whole image", "image", "photo", "picture",
                   "tum ", "her sey", "fotograf", "resim", "goruntu"],
}


def _detect_region(text: str) -> Optional[str]:
    for region, words in REGION_KEYWORDS.items():
        for w in words:
            # \b<word>\w*  matches the word and any agglutinative suffix
            # so "dudaklarini" → lips, "gozlerin" → eyes (Turkish-friendly).
            if re.search(r"\b" + re.escape(w) + r"\w*", text):
                return region
    return None


# Action verbs → canonical action key
ACTION_KEYWORDS = {
    "smooth":  ["smooth", "soften", "blur skin", "skin smooth", "yumusat",
                "yumusatici", "puruzsuz", "düzleştir", "duzlestir"],
    "sharpen": ["sharpen", "sharper", "crisp", "detail", "details",
                "keskinles", "keskin", "netlestir", "detayli"],
    "brighten":["bright", "brighten", "lift", "lighter", "lighten",
                "aydinlat", "isikla", "parlat"],
    "darken":  ["darken", "darker", "reduce light", "karart", "azalt isik"],
    "warm":    ["warm", "warmer", "warmth", "sicaklik", "sicakla", "sıcak"],
    "cool":    ["cool", "cooler", "soguk", "sogukla"],
    "colour":  ["colour", "color", "vibrant", "vibrance", "saturate",
                "pop", "renkli", "canli", "canlandir", "doygunluk"],
    "denoise": ["denoise", "noise", "grainy", "clean", "smooth noise",
                "gurultusu", "parazit", "temizle"],
    "restore": ["restore", "fix", "enhance", "repair", "improve",
                "iyilestir", "duzelt", "onar", "tamir"],
}


def _detect_action(text: str) -> Optional[str]:
    for action, words in ACTION_KEYWORDS.items():
        for w in words:
            if re.search(r"\b" + re.escape(w) + r"\w*", text):
                return action
    return None


# Live-filter triggers — must contain the keyword *and* the word "filter"
# (or a Turkish equivalent) to avoid stomping over scene presets.  E.g.
# "beauty filter" → BEAUTY, but "make it look beautiful" stays a preset.
FILTER_KEYWORDS = {
    "AI":      ["ai filter", "yapay zeka filtre", "ai filtresi"],
    "ENHANCE": ["enhance filter", "global filter", "enhance filtresi",
                "geli\u015ftir filtre", "gelistir filtre"],
    "BEAUTY":  ["beauty filter", "beauty mode", "skin filter",
                "guzellik filtre", "g\u00fczellik filtre", "cilt filtre"],
    "OFF":     ["filter off", "no filter", "remove filter", "stop filter",
                "filtreyi kapat", "filtre kapat", "filtresiz", "filtreyi kald\u0131r"],
}


def _detect_filter_name(text: str) -> Optional[str]:
    """Return one of OFF/BEAUTY/ENHANCE/AI iff text clearly asks for it."""
    for name, phrases in FILTER_KEYWORDS.items():
        for p in phrases:
            if p in text:
                return name
    return None


# Direct intent triggers (verbs that map to a single GUI command)
INTENT_TRIGGERS = {
    "save":     ["save", "export", "kaydet", "disa aktar"],
    "open":     ["open image", "open photo", "load image", "load photo",
                 "resim ac", "fotograf ac", "dosya ac"],
    "reset":    ["reset", "undo", "revert", "start over", "sifirla", "geri al",
                 "basa don"],
    "live":     ["live", "camera", "webcam", "kamera", "canli"],
    "capture":  ["capture", "snapshot", "shot", "take a photo", "cek",
                 "yakala", "fotograf cek"],
    "batch":    ["batch", "folder", "multiple files", "klasor", "toplu",
                 "coklu"],
    "compare":  ["compare", "before after", "before/after", "diff between",
                 "karsilastir", "kiyas"],
    "diff":     ["difference", "diff", "heatmap", "where changed",
                 "fark", "isi haritasi"],
    "original": ["original", "show original", "before", "orijinal", "kaynak",
                 "ham hali"],
    "restored": ["restored", "after", "result", "enhanced",
                 "iyilestirilmis", "duzeltilmis", "cikti", "sonuc"],
    "center":   ["only center", "center face", "center only",
                 "sadece merkez", "merkez yuz"],
    "all_faces":["all faces", "every face", "tum yuzler", "her yuz"],
    "status":   ["status", "explain", "what happened", "report",
                 "durum", "ne oldu", "anlat", "rapor"],
    "greet":    ["hi ", "hello", "hey", "merhaba", "selam"],
    "thanks":   ["thank", "thx", "tesekkur", "sagol", "sağol"],
    "help":     ["help", "what can you", "how do i", "yardim", "ne yapabilir",
                 "nasil kullan"],
}


def _has_any(text: str, words) -> bool:
    return any(w in text for w in words)


def _has_intent(text: str, key: str) -> bool:
    return _has_any(text, INTENT_TRIGGERS.get(key, []))


# Numeric fidelity / strength
_NUMERIC_RE = re.compile(
    r"(?:fidelity|strength|amount|level|etki|guc|seviye)\D{0,12}(\d{1,3})",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"(\d{1,3})\s*%")
_NUMBER_RE  = re.compile(r"\b(\d{1,3})\b")


# ── Parser ────────────────────────────────────────────────────────────

def _parse_segment(text: str, raw: str) -> Optional[Command]:
    """Parse a single clause (no compound splitting). Returns None if empty."""
    if not text.strip():
        return None

    # 1) greet / thanks → conversational
    if _has_intent(text, "greet"):
        return Command(intent="greet", raw_text=raw,
                       explanation="Hi! Tell me what you'd like to do — "
                                   "e.g. \"fix this old photo\" or "
                                   "\"smooth her skin a little\".")
    if _has_intent(text, "thanks"):
        return Command(intent="thanks", raw_text=raw,
                       explanation="You're welcome.")

    # 2) help
    if _has_intent(text, "help"):
        return Command(intent="help", raw_text=raw, explanation=_help_text())

    # 3) status / explain
    if _has_intent(text, "status"):
        return Command(intent="status", raw_text=raw,
                       explanation="Showing the latest pipeline report.")

    # 4) Region + action wins early — "smooth her skin a little" is more
    #    specific than the "skin" trigger of any preset.
    region = _detect_region(text)
    action = _detect_action(text)
    if region and action and action != "restore":
        mag = _magnitude(text)
        if mag is None:
            mag = 0.65 if action == "sharpen" else 0.55
        return Command(
            intent="tweak_region",
            params={"region": region, "action": action, "magnitude": mag},
            raw_text=raw,
            explanation="OK — {} the {} ({:.0%} strength).".format(
                _action_verb(action), _region_label(region), mag),
        )

    # 4b) Live-filter switch: "beauty filter", "ai filter on", "filter off"
    filt = _detect_filter_name(text)
    if filt is not None:
        return Command(
            intent="set_filter",
            params={"name": filt},
            raw_text=raw,
            explanation="Live filter -> {}".format(filt),
        )

    # 5) Scene preset triggers — match natural-language "looks"
    preset = find_preset_by_trigger(text)
    if preset is not None:
        return Command(
            intent="preset",
            params={"name": preset.name},
            raw_text=raw,
            explanation="Switching to '{}' look — {}".format(
                preset.name, preset.description),
        )

    # 6) Direct GUI triggers
    if _has_intent(text, "compare"):
        return Command(intent="view", params={"which": "compare"},
                       raw_text=raw, explanation="Showing before / after.")
    if _has_intent(text, "diff"):
        return Command(intent="view", params={"which": "diff"},
                       raw_text=raw, explanation="Showing the difference heatmap.")
    if _has_intent(text, "original"):
        return Command(intent="view", params={"which": "original"},
                       raw_text=raw, explanation="Showing the original.")
    if _has_intent(text, "restored"):
        return Command(intent="view", params={"which": "restored"},
                       raw_text=raw, explanation="Showing the restored result.")

    if _has_intent(text, "center"):
        return Command(intent="center_only", params={"flag": True},
                       raw_text=raw,
                       explanation="OK — only the centre face will be restored.")
    if _has_intent(text, "all_faces"):
        return Command(intent="center_only", params={"flag": False},
                       raw_text=raw,
                       explanation="OK — every detected face will be restored.")

    if _has_intent(text, "capture"):
        return Command(intent="capture", raw_text=raw,
                       explanation="Snapping a frame from the camera.")
    if _has_intent(text, "live"):
        return Command(intent="live", raw_text=raw,
                       explanation="Toggling live camera mode.")
    if _has_intent(text, "batch"):
        return Command(intent="batch", raw_text=raw,
                       explanation="Opening the batch picker.")
    if _has_intent(text, "save"):
        return Command(intent="save", raw_text=raw,
                       explanation="Bringing up Save…")
    if _has_intent(text, "open"):
        return Command(intent="open", raw_text=raw,
                       explanation="Bringing up Open…")
    if _has_intent(text, "reset"):
        return Command(intent="reset", raw_text=raw,
                       explanation="Resetting back to the original image.")

    # 6) Explicit fidelity number ("fidelity 40", "40%")
    m = _NUMERIC_RE.search(text) or _PERCENT_RE.search(text)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return Command(intent="set_fidelity", params={"pct": pct},
                       raw_text=raw,
                       explanation="Fidelity set to {}%.".format(pct))

    # 7) Magnitude alone for fidelity ("more", "less", "softer")
    mag = _magnitude(text)
    if mag is not None and ("ai" in text or "fidelity" in text or
                            "stronger" in text or "softer" in text or
                            "natural" in text or "guc" in text):
        if any(t in text for t in ("less ai", "softer", "natural",
                                   "subtle", "lighter")):
            return Command(intent="adjust_fidelity",
                           params={"delta": +int(mag * 30)}, raw_text=raw,
                           explanation="Toning down the AI a notch.")
        return Command(intent="adjust_fidelity",
                       params={"delta": -int(mag * 30)}, raw_text=raw,
                       explanation="Pushing more AI into the result.")

    # 8) Action verbs without region → fall back to global behaviours
    if action == "restore" or _has_any(text, ["enhance", "fix", "iyilestir",
                                              "duzelt", "uygula", "yap",
                                              "calistir", "apply"]):
        return Command(intent="restore", raw_text=raw,
                       explanation="Running enhancement now.")

    if action == "brighten":
        return Command(intent="preset", params={"name": "low_light"},
                       raw_text=raw,
                       explanation="Lifting shadows and brightening the photo.")
    if action == "colour":
        return Command(intent="preset", params={"name": "vivid"},
                       raw_text=raw,
                       explanation="Going for a punchy, colourful grade.")
    if action in ("denoise", "warm", "cool", "darken"):
        mag = _magnitude(text) or 0.6
        return Command(
            intent="tweak_region",
            params={"region": "all", "action": action, "magnitude": mag},
            raw_text=raw,
            explanation="OK — {} the whole image ({:.0%}).".format(
                _action_verb(action), mag),
        )

    return None


def parse(user_text: str) -> Command:
    """
    Parse free-form text into a Command. Compound sentences (split by
    'and', '+', ',', ';' and Turkish 've') yield a `compound` Command
    whose `intents` list contains the individual Commands.

    This is what makes the chat feel natural: a single sentence such as
    "smooth her skin a lot and sharpen the eyes" produces *two* actions.
    """
    if not user_text or not user_text.strip():
        return Command(intent="unknown",
                       explanation="Type something — e.g. \"fix this old photo\".",
                       raw_text=user_text or "")

    raw   = user_text
    text  = _normalise(user_text)

    # Split on conjunctions while keeping commas as soft separators.
    parts = re.split(r"\s+(?:and|then|plus|ve)\s+|[,;]\s*|\s\+\s", text)
    parts = [p for p in (p.strip() for p in parts) if p]

    cmds: List[Command] = []
    for p in parts:
        c = _parse_segment(p, raw)
        if c is not None:
            cmds.append(c)

    if not cmds:
        return Command(
            intent="unknown", raw_text=raw,
            explanation=("I'm not sure what you mean. Try \"restore\", "
                         "\"smooth her skin a little\", \"fix this old photo\", "
                         "\"compare\" or \"help\"."),
        )

    if len(cmds) == 1:
        return cmds[0]

    summary = " · ".join(c.explanation for c in cmds if c.explanation)
    return Command(intent="compound", raw_text=raw,
                   explanation=summary, intents=cmds)


# ── Helpers for human-friendly ack strings ───────────────────────────

def _action_verb(action: str) -> str:
    return {
        "smooth":   "softening",
        "sharpen":  "sharpening",
        "brighten": "brightening",
        "darken":   "darkening",
        "warm":     "warming",
        "cool":     "cooling",
        "colour":   "saturating colours of",
        "denoise":  "cleaning noise on",
        "restore":  "restoring",
    }.get(action, action)


def _region_label(region: str) -> str:
    return {
        "skin":       "skin",
        "eyes":       "eyes",
        "lips":       "lips",
        "brows":      "brows",
        "nose":       "nose",
        "hair":       "hair",
        "face":       "face",
        "background": "background",
        "all":        "whole image",
    }.get(region, region)


def _help_text() -> str:
    return (
        "I understand natural language. Try:\n"
        "  - \"fix this old photo\"        -> heavy AI restoration\n"
        "  - \"make it look professional\" -> magazine grade\n"
        "  - \"smooth her skin a little, sharpen the eyes\"\n"
        "  - \"brighten this photo\"       -> low-light preset\n"
        "  - \"make it pop\" or \"vivid\"  -> punchy colour\n"
        "  - \"natural\" or \"subtle\"     -> minimal-touch\n"
        "  - \"fidelity 40\" or \"40%\"\n"
        "  - \"compare\", \"diff\", \"reset\", \"save\"\n"
        "Live mode (after \"live\"):\n"
        "  - \"beauty filter\", \"ai filter\", \"enhance filter\", \"filter off\"\n"
        "  - \"capture\" to snapshot the current frame\n"
        "  - \"batch\" to bulk-process a folder"
    )


# ── Executor (GUI glue) ───────────────────────────────────────────────

class ChatExecutor:
    """
    Calls MainWindow methods for a parsed Command. Supports compound
    commands by dispatching each child intent in order.
    """

    VIEW_MAP = {"original": 0, "restored": 1, "compare": 2, "diff": 3}

    @staticmethod
    def dispatch(cmd: Command, w) -> str:
        """Execute `cmd` against MainWindow `w`. Never raises."""
        try:
            if cmd.intent == "compound":
                outs = []
                for child in cmd.intents:
                    outs.append(ChatExecutor.dispatch(child, w))
                return "  · ".join(o for o in outs if o)

            i, p = cmd.intent, cmd.params

            if i == "set_fidelity":
                pct = int(p.get("pct", 50))
                w._sldFW.setValue(pct)
                return "Fidelity → {}%.".format(pct)

            if i == "adjust_fidelity":
                cur = w._sldFW.value()
                new = max(0, min(100, cur + int(p.get("delta", 0))))
                w._sldFW.setValue(new)
                return "Fidelity {}% → {}%.".format(cur, new)

            if i == "preset":
                return _apply_preset(w, p.get("name", ""))

            if i == "tweak_region":
                return _apply_tweak(w, p)

            if i == "center_only":
                flag = bool(p.get("flag", False))
                if hasattr(w, "_ckCenter"):
                    w._ckCenter.setChecked(flag)
                return "Centre-only: {}.".format("on" if flag else "off")

            if i == "view":
                which = str(p.get("which", "original"))
                idx = ChatExecutor.VIEW_MAP.get(which, 0)
                if hasattr(w, "_vbtns") and idx < len(w._vbtns):
                    w._vbtns[idx].setChecked(True)
                    w._chg_view(idx)
                return "View: {}.".format(which)

            if i == "set_filter":
                name = str(p.get("name", "OFF")).upper()
                names = ["OFF", "BEAUTY", "ENHANCE", "AI"]
                if name not in names:
                    return "Unknown filter '{}'.".format(name)
                idx = names.index(name)
                if hasattr(w, "_fbtns") and idx < len(w._fbtns):
                    w._fbtns[idx].setChecked(True)
                    w._set_filter(idx)
                return "Live filter -> {}.".format(name)

            if i == "restore":
                if w._orig is None:
                    return "Load a photo first (drag-drop or 'open')."
                w._restore()
                return "Running enhancement…"

            if i == "reset":
                w._reset()
                return "Reverted to the original."

            if i == "save":
                if w._rest is None:
                    return "Nothing to save yet — try 'restore' first."
                w._save()
                return "Save dialog opened."

            if i == "open":
                w._load()
                return "Open dialog shown."

            if i == "live":
                w._bLive.setChecked(not w._bLive.isChecked())
                w._toggle_live(w._bLive.isChecked())
                return "Live mode: {}.".format(
                    "on" if w._bLive.isChecked() else "off")

            if i == "capture":
                if not w._cam.is_open:
                    return "Camera not active — say 'live' first."
                w._capture_shot()
                return "Snapshot captured."

            if i == "batch":
                w._batch()
                return "Batch picker opened."

            if i == "status":
                return _status_summary(w)

            if i in ("greet", "thanks", "help"):
                return cmd.explanation

            return cmd.explanation or "Sorry, I didn't catch that."

        except Exception as ex:
            return "Sorry — that command failed: {}".format(ex)


# ── Executor helpers ──────────────────────────────────────────────────

def _apply_preset(w, name: str) -> str:
    """Apply a ScenePreset by writing fidelity slider + cached preset."""
    from scene_presets import find_preset_by_name
    preset = find_preset_by_name(name)
    if preset is None:
        return "Preset '{}' not found.".format(name)

    w._active_preset = preset                # MainWindow reads this on restore
    pct = int(round(preset.fidelity * 100))
    w._sldFW.setValue(pct)

    if w._orig is not None:
        w._restore()
        return ("Applied '{}' preset (fidelity {}%) — {}").format(
            preset.name, pct, preset.description)
    return ("Saved '{}' preset — load a photo and I'll apply it. "
            "({})".format(preset.name, preset.description))


def _apply_tweak(w, p: Dict) -> str:
    """
    Map a (region, action, magnitude) tuple to a preset-style override
    on the active preset (creating a custom one if none is active).
    """
    region = str(p.get("region", "all"))
    action = str(p.get("action", "restore"))
    mag    = float(p.get("magnitude", 0.5))

    base = getattr(w, "_active_preset", None)
    if base is None:
        from scene_presets import PRESETS
        base = PRESETS["portrait"]

    from copy import deepcopy
    custom = deepcopy(base)
    custom.name = "custom"
    custom.description = "Custom — {} the {}".format(
        _action_verb(action), _region_label(region))

    # Region-specific knobs
    if region in ("skin", "face"):
        if action == "smooth":   custom.skin_amount  = mag
        if action == "sharpen":  custom.eyes_sharpen = max(custom.eyes_sharpen, mag)
        if action == "restore":  custom.skin_amount  = mag
    elif region == "eyes":
        if action == "sharpen":  custom.eyes_sharpen = mag
        if action == "brighten": custom.eyes_bright  = mag
    elif region == "lips":
        if action == "colour":   custom.lips_vibrance = mag
        if action == "warm":     custom.lips_warmth   = mag
    elif region == "brows":
        custom.brows_amount = mag
    elif region == "nose":
        custom.nose_amount = mag

    # Global-level actions
    if action == "brighten":
        custom.shadow_lift = max(custom.shadow_lift or 0.0, mag)
        custom.force_global = True
    if action == "darken":
        custom.shadow_lift = 1.0 - mag
        custom.force_global = True
    if action == "warm":
        custom.white_balance = 0.4 - 0.3 * mag
        custom.force_global = True
    if action == "cool":
        custom.white_balance = 0.6 + 0.3 * mag
        custom.force_global = True
    if action == "colour":
        custom.vibrance = mag
        custom.force_global = True
    if action == "denoise":
        custom.bilateral = mag
        custom.force_global = True

    w._active_preset = custom
    if w._orig is not None:
        w._restore()
        return "Tweaked: {} the {} ({:.0%}).".format(
            _action_verb(action), _region_label(region), mag)
    return "Saved tweak — load a photo to apply."


def _status_summary(w) -> str:
    """Describe the latest pipeline result for 'status' / 'explain'."""
    if not getattr(w, "_lastResult", None):
        return "Nothing has been processed yet."
    r = w._lastResult
    parts = []
    if getattr(r, "plan", None):
        ran = ", ".join(s.layer.value for s in r.plan.steps if s.run)
        parts.append("Layers run: " + (ran or "none"))
    if getattr(r, "guard_reports", None):
        scores = ["{}={:.0f}".format(k, v.trust)
                  for k, v in r.guard_reports.items()]
        parts.append("Trust: " + ", ".join(scores))
    if getattr(r, "metrics", None):
        parts.append("PSNR={:.1f} dB · SSIM={:.3f}".format(
            r.metrics.psnr, r.metrics.ssim))
    return "  ·  ".join(parts) or "Processed."
