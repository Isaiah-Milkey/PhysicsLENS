"""Stage + cost helpers for pipelines.

The backend registry tags every pipeline with a cost `badge`
(cheap/medium/expensive/output) but NOT a stage number — stage is encoded in
the id prefix (``s1_``/``s2_``/``s3_``/``s4_``), a convention every registered
pipeline follows. These helpers centralize that so the rest of the MCP can
reason about how deep (stage) and how expensive (budget) a pipeline is without
re-deriving it everywhere.
"""
import re

# Ascending cost. A `budget` selects every pipeline whose tier is <= the budget.
COST_ORDER: dict[str, int] = {"cheap": 1, "medium": 2, "expensive": 3, "output": 4}

_STAGE_RE = re.compile(r"^s(\d+)_")

# Matches CLAUDE.md's stage table / the frontend's STAGES array names.
STAGE_NAMES: dict[int, str] = {
    1: "Screening", 2: "Differential Diagnosis",
    3: "Specialist Evaluation", 4: "Final Diagnosis",
}


def stage_of(pipeline_id: str) -> int | None:
    """Stage number (1-4) parsed from an ``sN_`` id prefix, or None if the id
    doesn't follow the convention."""
    m = _STAGE_RE.match(pipeline_id or "")
    return int(m.group(1)) if m else None


def cost_rank(badge: str) -> int:
    """Sortable rank for a cost badge; unknown badges sort last."""
    return COST_ORDER.get((badge or "").lower(), 99)
