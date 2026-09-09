"""Physics-reasoning taxonomy and frame-filename conventions.

Edit TAXONOMY here if the table changes; every other module reads from it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class Domain:
    code: str
    domain: str
    matter_state: str
    type: str  # Single | Cross-matter | Implicit | Compound

    def as_dict(self) -> dict:
        return asdict(self)


TAXONOMY: dict[str, Domain] = {
    d.code: d
    for d in [
        Domain("RC",  "Rigid Contact & Collision",       "Solid",                    "Single"),
        Domain("FR",  "Friction & Sliding",              "Solid",                    "Single"),
        Domain("SS",  "Support & Stability",             "Solid",                    "Single"),
        Domain("MI",  "Momentum & Inertia",              "Solid",                    "Single"),
        Domain("FD",  "Fluid Dynamics",                  "Liquid",                   "Single"),
        Domain("DM",  "Deformable / Soft Material",      "Solid (soft)",             "Single"),
        Domain("GP",  "Gas & Particulate",               "Gas",                      "Single"),
        Domain("SL",  "Solid-Liquid Interaction",        "Solid + Liquid",           "Cross-matter"),
        Domain("LG",  "Liquid-Gas Interaction",          "Liquid + Gas",             "Cross-matter"),
        Domain("GS",  "Gas-Solid Interaction",           "Gas + Solid",              "Cross-matter"),
        Domain("SLG", "Solid-Liquid-Gas Interaction",    "Solid + Liquid + Gas",     "Cross-matter"),
        Domain("BY",  "Buoyancy & Density",              "Solid + Liquid",           "Implicit"),
        Domain("TH",  "Thermal State",                   "Liquid / Gas",             "Implicit"),
        Domain("MG",  "Magnetism",                       "Solid",                    "Implicit"),
        Domain("CX",  "Compound",                        "Multi",                    "Compound"),
    ]
}

# Domain-specific nudges for the VLM. Purely advisory: they steer what the model
# pays attention to, they do not add fields to the output.
FOCUS_HINTS: dict[str, str] = {
    "RC":  "contact points, impact direction, restitution, what strikes what and where",
    "FR":  "surface textures, contact area, incline angle, whether motion is impending or ongoing",
    "SS":  "base of support, centre of mass relative to that base, stacking order, toppling axis",
    "MI":  "mass distribution, speed and direction of motion, what happens when motion is arrested",
    "FD":  "flow direction, container geometry, free surface, pouring/draining rate",
    "DM":  "compliance of the material, where load is applied, expected deformation mode",
    "GP":  "dispersion, plume/jet direction, entrainment, settling of particulates",
    "SL":  "wetting, displacement, immersion, splashing at the solid-liquid boundary",
    "LG":  "bubbles, evaporation, surface disturbance, gas trapped in or above liquid",
    "GS":  "airflow over/around the solid, drag, whether the solid is light enough to be moved",
    "SLG": "the three-phase contact and how each phase mediates the others",
    "BY":  "relative density, submerged fraction, whether the object floats, sinks, or hovers",
    "TH":  "temperature cues (steam, ice, condensation), phase change, heat transfer direction",
    "MG":  "which objects are ferromagnetic, attraction/repulsion, gap between magnet and object",
    "CX":  "the ordered chain of physical events; identify each stage and its dependency",
}

OBSERVABLE = "observable"
UNOBSERVABLE = "unobservable"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# <CODE>_<ID>_<obs|unobs>.<ext>   e.g. RC_001_obs.jpg, MG_012_unobs.png
# Tokens are case-insensitive and both "obs"/"observable" spellings are accepted.
_OBS_TOKENS = {
    "obs": OBSERVABLE, "o": OBSERVABLE, "observable": OBSERVABLE,
    "unobs": UNOBSERVABLE, "u": UNOBSERVABLE, "unobservable": UNOBSERVABLE,
    "hidden": UNOBSERVABLE,
}

_FILENAME_RE = re.compile(
    r"^(?P<code>[A-Za-z]{2,3})[_-](?P<id>[A-Za-z0-9]+)[_-](?P<obs>[A-Za-z]+)$"
)


class FilenameError(ValueError):
    """Raised when a frame filename does not follow the naming convention."""


@dataclass(frozen=True)
class FrameSpec:
    path: str
    sample_id: str      # e.g. "RC_001"
    domain: Domain
    observability: str  # observable | unobservable

    @property
    def is_unobservable(self) -> bool:
        return self.observability == UNOBSERVABLE


def parse_frame_filename(path: str) -> FrameSpec:
    """Derive taxonomy code and observability from a frame's filename.

    Expected form: <CODE>_<ID>_<obs|unobs>.<ext>, e.g. ``BY_004_unobs.jpg``.
    """
    stem, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() not in IMAGE_EXTS:
        raise FilenameError(f"{path}: unsupported image extension {ext!r}")

    m = _FILENAME_RE.match(stem)
    if not m:
        raise FilenameError(
            f"{path}: expected <CODE>_<ID>_<obs|unobs>{ext}, e.g. RC_001_obs{ext}"
        )

    code = m.group("code").upper()
    if code not in TAXONOMY:
        raise FilenameError(
            f"{path}: unknown taxonomy code {code!r}; known codes: {', '.join(TAXONOMY)}"
        )

    obs = _OBS_TOKENS.get(m.group("obs").lower())
    if obs is None:
        raise FilenameError(
            f"{path}: unknown observability token {m.group('obs')!r}; use 'obs' or 'unobs'"
        )

    return FrameSpec(
        path=path,
        sample_id=f"{code}_{m.group('id')}",
        domain=TAXONOMY[code],
        observability=obs,
    )


def find_frames(frames_dir: str, recursive: bool = True) -> list[str]:
    """Return every image file under ``frames_dir``, sorted, skipping dotfiles."""
    out: list[str] = []
    if recursive:
        for root, dirs, files in os.walk(frames_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            out += [
                os.path.join(root, f)
                for f in files
                if not f.startswith(".") and os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]
    else:
        out = [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if not f.startswith(".") and os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
    return sorted(out)
