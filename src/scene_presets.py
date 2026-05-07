"""
Scene Presets — declarative photo styles for OptiLumen.

Each preset is a fully-specified set of pipeline parameters (fidelity,
region amounts, global stack flags) that produces a recognisable look
for a specific scenario:

    portrait     — single subject, balanced restoration + cosmetic
    group        — multiple subjects, conservative cosmetic
    old_photo    — heavy GFPGAN, denoise, contrast restoration
    magazine     — high-fashion glossy look (vibrant, sharp, smoothed)
    natural      — minimal AI touch, mostly classical
    low_light    — global lift + denoise + face restore
    vivid        — punchy colour grade
    document     — text/scan focused, no face cosmetic

The presets module is fully data-driven; chat_commands maps natural
language triggers ("make it look professional", "fix this old photo")
to preset names, and the GUI's restore loop reads the active preset
to override parameters.

This is what makes the system *adaptive across diverse scenarios* —
the same pipeline produces very different looks depending on which
preset is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Preset definition ────────────────────────────────────────────────

@dataclass
class ScenePreset:
    """One named photo style."""
    name:          str
    description:   str
    fidelity:      float = 0.5            # 0=full AI, 1=passthrough

    # Region (face cosmetic) overrides
    skin_amount:   float = 0.55
    nose_amount:   float = 0.35
    brows_amount:  float = 0.45
    eyes_sharpen:  float = 0.70
    eyes_bright:   float = 0.25
    lips_vibrance: float = 0.55
    lips_warmth:   float = 0.30

    # Global enhancer overrides (None = let GlobalEnhancer pick its default)
    white_balance: Optional[float] = None
    shadow_lift:   Optional[float] = None
    bilateral:     Optional[float] = None
    clahe_clip:    Optional[float] = None
    hdr:           Optional[float] = None
    sharpen:       Optional[float] = None
    vibrance:      Optional[float] = None
    film_grade:    Optional[float] = None

    # Pipeline-level overrides
    force_global:  bool = False           # run GlobalEnhance even if profile says skip
    force_general: bool = False           # run Real-ESRGAN even if image is "fine"
    skip_global:   bool = False           # never run GlobalEnhance
    skip_general:  bool = False           # never run Real-ESRGAN

    # Natural-language triggers (lowercase substrings, English + Turkish)
    triggers: List[str] = field(default_factory=list)


# ── Library ──────────────────────────────────────────────────────────

PRESETS: Dict[str, ScenePreset] = {

    # 1) Portrait — the default for single-subject face photos
    "portrait": ScenePreset(
        name="portrait",
        description="Balanced face restoration + subtle cosmetic.",
        fidelity=0.50,
        skin_amount=0.55, eyes_sharpen=0.65, lips_vibrance=0.55,
        triggers=[
            "portrait", "headshot", "selfie", "single person",
            "portre", "vesikalik", "vesikalık",
        ],
    ),

    # 2) Group photo — many faces, keep it natural
    "group": ScenePreset(
        name="group",
        description="Multiple faces — conservative cosmetic, balanced colour.",
        fidelity=0.55,
        skin_amount=0.40, nose_amount=0.20, brows_amount=0.30,
        eyes_sharpen=0.50, lips_vibrance=0.40, lips_warmth=0.20,
        triggers=[
            "group", "family", "team photo", "everyone", "multiple",
            "grup", "aile", "kalabalik", "kalabalık", "topluluk",
        ],
    ),

    # 3) Old photo — heavy AI restoration
    "old_photo": ScenePreset(
        name="old_photo",
        description="Heavy AI restoration for damaged / faded / scanned photos.",
        fidelity=0.20,                    # max AI lift
        skin_amount=0.65, nose_amount=0.45, brows_amount=0.55,
        eyes_sharpen=0.85, lips_vibrance=0.65, lips_warmth=0.40,
        clahe_clip=4.5, sharpen=0.85, vibrance=0.70,
        force_general=True, force_global=True,
        triggers=[
            "old photo", "vintage photo", "restore old", "fix old", "scanned",
            "faded", "damaged", "fix this old", "fix this photo", "antique",
            "eski fotograf", "eski fotoğraf", "yipranmis", "yıpranmış",
            "tarama", "vintage", "antika",
        ],
    ),

    # 4) Magazine — high-end glossy look
    "magazine": ScenePreset(
        name="magazine",
        description="High-fashion magazine look — punchy, smooth, sharp.",
        fidelity=0.40,
        skin_amount=0.75, nose_amount=0.45, brows_amount=0.55,
        eyes_sharpen=0.85, eyes_bright=0.40, lips_vibrance=0.75, lips_warmth=0.35,
        vibrance=0.80, sharpen=0.80, film_grade=0.65,
        triggers=[
            "magazine", "professional", "studio", "fashion", "glossy",
            "editorial", "vogue", "instagram", "ig look", "make it look pro",
            "professional headshot", "modern", "modernist",
            "dergi", "moda", "stüdyo", "studyo", "profesyonel",
        ],
    ),

    # 5) Natural — minimum AI touch
    "natural": ScenePreset(
        name="natural",
        description="Minimal touch — barely-there cleanup, true to original.",
        fidelity=0.75,                    # mostly original
        skin_amount=0.30, nose_amount=0.20, brows_amount=0.25,
        eyes_sharpen=0.40, lips_vibrance=0.30, lips_warmth=0.15,
        skip_global=True,
        triggers=[
            "natural", "subtle", "barely", "just clean", "minimal",
            "soft", "gentle", "light touch", "no filter",
            "doğal", "dogal", "hafif", "az dokunma", "filtresiz",
        ],
    ),

    # 5b) Natural+ — default "phone-like" enhancement (safe but visible)
    "natural_plus": ScenePreset(
        name="natural_plus",
        description="Default look — natural but visibly cleaner/sharper, phone-like.",
        fidelity=0.65,
        skin_amount=0.40, nose_amount=0.25, brows_amount=0.30,
        eyes_sharpen=0.50, eyes_bright=0.20,
        lips_vibrance=0.40, lips_warmth=0.20,
        # allow global stack, but keep it mild via config mapping defaults
        # (do not force; DecisionEngine may still skip if the image looks fine)
        triggers=[
            "natural+", "natural plus", "default look", "phone look", "samsung",
            "natural artı", "dogal arti", "doğal artı",
            "varsayilan", "varsayılan",
        ],
    ),

    # 5c) Modern touch — always apply a *mild* global pass (safe phone look)
    #
    # Use this when users explicitly want "make it look better" even if the
    # image is already clean. Guardrails still apply, so over-processing
    # is blended/rejected automatically.
    "modern_touch": ScenePreset(
        name="modern_touch",
        description="Always-on mild global polish (natural phone-style).",
        fidelity=0.70,
        skin_amount=0.35, nose_amount=0.22, brows_amount=0.28,
        eyes_sharpen=0.45, eyes_bright=0.18,
        lips_vibrance=0.35, lips_warmth=0.18,
        force_global=True,
        # Mild global overrides (preset maps clahe_clip/5 → clahe_strength)
        white_balance=0.40,
        shadow_lift=0.25,
        bilateral=0.20,
        clahe_clip=1.5,
        hdr=0.20,
        sharpen=0.25,
        vibrance=0.20,
        film_grade=0.10,
        triggers=[
            "modern touch", "polish", "make it better", "enhance", "remaster",
            "phone enhance", "samsung ai", "iphone look",
            "iyilestir", "iyileştir", "parlat", "duzelt", "düzelt",
            "remaster", "guzellestir", "güzelleştir",
        ],
    ),

    # 6) Low-light — shadow lift + denoise
    "low_light": ScenePreset(
        name="low_light",
        description="Lift shadows, suppress noise, restore faces in dim photos.",
        fidelity=0.35,
        skin_amount=0.55, eyes_sharpen=0.75, eyes_bright=0.45,
        shadow_lift=0.85, bilateral=0.75, clahe_clip=3.0, vibrance=0.55,
        force_global=True, force_general=True,
        triggers=[
            "low light", "dark photo", "night", "underexposed", "too dark",
            "shadow", "brighten this", "lighten",
            "karanlik", "karanlık", "az isik", "az ışık", "gece", "loş",
            "karanliği aç", "aydınlat", "gündüz gibi", "aydinlat",
        ],
    ),

    # 7) Vivid — punchy colour-forward grade
    "vivid": ScenePreset(
        name="vivid",
        description="Punchy, vibrant, saturated colour grade.",
        fidelity=0.45,
        skin_amount=0.50, lips_vibrance=0.85, lips_warmth=0.45,
        vibrance=0.90, sharpen=0.65, film_grade=0.55,
        triggers=[
            "vivid", "vibrant", "punchy", "colourful", "colorful", "saturated",
            "pop", "make it pop",
            "canli", "canlı", "renkli", "doygun",
        ],
    ),

    # 8) Document — text/scan, no face cosmetic
    "document": ScenePreset(
        name="document",
        description="Text / scanned document — no face cosmetic, sharpen + denoise.",
        fidelity=0.30,
        skin_amount=0.0, nose_amount=0.0, brows_amount=0.0,
        eyes_sharpen=0.0, lips_vibrance=0.0,
        clahe_clip=4.5, sharpen=0.90, vibrance=0.0,
        skip_global=False, force_general=True,
        triggers=[
            "document", "scan", "text", "paper", "receipt", "id card", "license",
            "doküman", "dokuman", "belge", "tarama", "yazı", "metin", "kimlik",
        ],
    ),
}


def find_preset_by_name(name: str) -> Optional[ScenePreset]:
    """Exact lookup by preset key."""
    if not name:
        return None
    return PRESETS.get(name.strip().lower())


def find_preset_by_trigger(text: str) -> Optional[ScenePreset]:
    """
    Match user text against every preset's trigger list. Returns the
    *most specific* preset (longest matched trigger wins, so
    "old photo" beats just "photo").
    """
    if not text:
        return None
    t = text.strip().lower()

    best: Optional[ScenePreset] = None
    best_len = 0
    for preset in PRESETS.values():
        for trigger in preset.triggers:
            if trigger in t and len(trigger) > best_len:
                best = preset
                best_len = len(trigger)
    return best


def list_presets() -> List[ScenePreset]:
    return list(PRESETS.values())
