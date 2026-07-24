"""Pipeline scope selection (budget/stages/explicit ids) and Stage 2 -> Stage 3
hypothesis routing — mirrors the app's frontend batch logic
(`routedSpecialistsFor` / `SPECIALIST_PIPES` in index.html) so an MCP-driven
evaluation behaves the same way a batch run in the UI would.
"""
from .taxonomy import cost_rank, stage_of

# Named budget -> ceiling cost badge (inclusive). "thorough" ceilings at
# "output" (the highest tier), so it deliberately includes everything, not
# just the expensive Stage 3 specialists — no special-casing s4_report.
BUDGET_CEILING_BADGE: dict[str, str] = {
    "cheap": "cheap", "standard": "medium", "thorough": "output",
}

# Stage-2 hypothesis "specialist" id -> Stage-3 pipeline id. Mirrors
# `SPECIALISTS` in physics_hypothesis_generator.py / `SPECIALIST_PIPES` in the
# frontend. Keep in sync if a new specialist pipeline is added.
SPECIALIST_PIPES: dict[str, str] = {
    "collision": "s3_collision", "gravity": "s3_gravity", "momentum": "s3_momentum",
    "friction": "s3_friction", "deformation": "s3_deformation", "fluid": "s3_fluid",
    "causality": "s3_causality",
}


def select_pipelines(catalog: list[dict], *, pipelines: list[str] | None = None,
                     stages: list[int] | None = None,
                     budget: str | None = None) -> list[dict]:
    """Resolve a run scope to an ordered (by stage, then cost) list of live
    pipeline dicts. Precedence: explicit `pipelines` > `stages` > `budget`
    (defaults to "standard" if none given). Stub and paired-video pipelines
    are always excluded — modular: any pipeline's own `dummy`/`requires_pair`/
    `badge` flags determine eligibility, no per-pipeline special-casing here.
    """
    live = {p["id"]: p for p in catalog
            if not p.get("dummy") and not p.get("requires_pair")}

    if pipelines:
        missing = [pid for pid in pipelines if pid not in live]
        if missing:
            raise ValueError(
                f"Unknown or unavailable pipeline id(s): {', '.join(missing)}")
        chosen = [live[pid] for pid in pipelines]
    elif stages:
        wanted = set(stages)
        chosen = [p for p in live.values() if stage_of(p["id"]) in wanted]
    else:
        badge = BUDGET_CEILING_BADGE.get(budget or "standard")
        if badge is None:
            raise ValueError(
                f"Unknown budget '{budget}'. Use one of: "
                + ", ".join(BUDGET_CEILING_BADGE))
        ceiling = cost_rank(badge)
        chosen = [p for p in live.values() if cost_rank(p.get("badge", "")) <= ceiling]

    chosen.sort(key=lambda p: (stage_of(p["id"]) if stage_of(p["id"]) is not None else 99,
                               cost_rank(p.get("badge", ""))))
    return chosen


def routed_specialists(hyp_result: dict, catalog_by_id: dict[str, dict],
                       top_n: int) -> list[dict]:
    """Given the Hypothesis Generator's aggregated `result` payload (its
    `hypotheses` list, already confidence-sorted descending), return the
    top-N distinct Stage 3 specialist pipeline dicts it recommends that are
    actually live in the catalog."""
    hyps = hyp_result.get("hypotheses") or []
    out: list[dict] = []
    seen: set[str] = set()
    for h in hyps:
        pid = SPECIALIST_PIPES.get(str(h.get("specialist", "")).lower())
        if not pid or pid in seen:
            continue
        p = catalog_by_id.get(pid)
        if not p or p.get("dummy"):
            continue
        out.append(p)
        seen.add(pid)
        if len(out) >= top_n:
            break
    return out
