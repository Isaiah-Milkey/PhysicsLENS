# One package per model family — see README.md in this folder for the contract.
# Helpers shared by every family live here.
import os
import re
import shutil
from pathlib import Path

# huggingface_hub downloads into $HF_HOME; report cache/disk from the same
# place. The fallback is Som's shared cache (readable but not writable by
# everyone — set HF_HOME to your own dir to download new models).
HF_ROOT = os.environ.get("HF_HOME", "/data/ssagar6/hf_cache")


def is_cached(repo: str) -> bool:
    """True if the repo already has a materialised snapshot in HF_HOME."""
    d = Path(HF_ROOT) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    return d.is_dir() and any(p.is_dir() for p in d.iterdir())


def free_gb(path=HF_ROOT) -> float:
    return shutil.disk_usage(path).free / 2**30


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:n].rstrip("-") or "clip")


# Clip length is specified in SECONDS, uniformly across families; each family
# converts to its checkpoint's frame count. 5 s is the benchmark default.
DEFAULT_SECONDS = 5.0


def frames_for(seconds: float, fps: float) -> int:
    """Nearest frame count to seconds*fps satisfying the n = 4k+1 constraint
    the video VAEs impose (temporal compression 4x plus the anchor frame)."""
    return 4 * max(1, round((seconds * fps - 1) / 4)) + 1
