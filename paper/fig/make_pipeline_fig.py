"""
Figure 3 — evidence flow through the PhysicsLENS diagnostic pipeline.

Drawn to match the implementation (audited against backend/ on 2026-09-24),
not the design document:
  * Stage-1 detectors do NOT write to the evidence store; their timestamped
    signals reach the Stage-2 event localizer through the orchestrator.
  * The Stage-1 VLM suspicion score reaches only the Stage-4 report.
  * Stages 2 and 3 write to the store; Stage 3 and Stage 4 read from it.
  * Routing = top-4 hypotheses (max of heuristic prior and VLM triage) -> top-3
    specialists run; there is no confidence threshold or cost gate.
  * Gates drawn in red are the ones that exist in code.

python paper/fig/make_pipeline_fig.py  ->  paper/fig/pipeline.pdf (+ .png preview)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

INK = "#1f2328"
MUTE = "#5a6270"
GATE = "#b3261e"
C = {1: "#e8f0fb", 2: "#e9f5ec", 3: "#fdf2e3", 4: "#f3ecf9", "store": "#f1f3f5"}
E = {1: "#4a78c2", 2: "#3f8f55", 3: "#c07d1c", 4: "#7c55b3", "store": "#8a929c"}

fig, ax = plt.subplots(figsize=(11.6, 5.2))
ax.set_xlim(0, 116)
ax.set_ylim(0, 52)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.0, r=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))


def text(x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("color", INK)
    ax.text(x, y, s, zorder=5, **kw)


def arrow(x0, y0, x1, y1, color=INK, lw=1.1, ls="-", rad=0.0, head=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_length={head/10},head_width={head/20}",
                                 mutation_scale=10, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


def gate(x, y, label, ha="center"):
    ax.plot([x], [y], marker="D", ms=5.5, color=GATE, zorder=6)
    text(x + (1.2 if ha == "left" else 0), y - 1.9, label, fontsize=6.2, color=GATE,
         ha=ha, style="italic")


# ── stage columns ────────────────────────────────────────────────────────────
X = {1: 1.0, 2: 31.0, 3: 61.0, 4: 91.0}
W, Y0, H = 25.0, 17.5, 32.5
for s, title in [(1, "Stage 1 · Screening"), (2, "Stage 2 · Localization &\ndifferential diagnosis"),
                 (3, "Stage 3 · Specialists"), (4, "Stage 4 · Report")]:
    box(X[s], Y0, W, H, C[s], E[s], lw=1.3)
    text(X[s] + W / 2, Y0 + H - 2.6, title, fontsize=8.6, weight="bold", color=E[s])

# Stage 1 contents
x1 = X[1] + 1.2
for i, (name, desc) in enumerate([
        ("Temporal smoothness", "keypoint acceleration breaks"),
        ("Optical-flow irregularity", "flow spikes, incoherence"),
        ("Embedding biomarkers", "DINOv2 latent jumps"),
        ("Camera motion", "global jitter vs. object motion")]):
    yy = Y0 + H - 8.2 - i * 4.6
    box(x1, yy - 1.7, W - 2.4, 3.6, "white", E[1], lw=0.7, r=0.6)
    text(x1 + 1.0, yy + 0.55, name, ha="left", fontsize=7.2, weight="bold")
    text(x1 + 1.0, yy - 0.8, desc, ha="left", fontsize=6.4, color=MUTE)
yy = Y0 + 3.4
box(x1, yy - 1.7, W - 2.4, 3.6, "white", E[1], lw=0.7, r=0.6, ls=(0, (3, 2)))
text(x1 + 1.0, yy + 0.55, "VLM suspicion", ha="left", fontsize=7.2, weight="bold")
text(x1 + 1.0, yy - 0.8, "whole-clip score + violations", ha="left", fontsize=6.4, color=MUTE)

# Stage 2 contents
x2 = X[2] + 1.2
s2 = [("Object tracker", "VLM-named SAM 3 masks / LK tracks"),
      ("Trajectory extractor", "velocity, acceleration, contacts"),
      ("Event localizer", "fused signals → time windows"),
      ("Hypothesis generator", "max(heuristic prior, VLM triage)")]
for i, (name, desc) in enumerate(s2):
    yy = Y0 + H - 8.2 - i * 5.6
    box(x2, yy - 1.9, W - 2.4, 4.0, "white", E[2], lw=0.7, r=0.6)
    text(x2 + 1.0, yy + 0.65, name, ha="left", fontsize=7.2, weight="bold")
    text(x2 + 1.0, yy - 0.85, desc, ha="left", fontsize=6.4, color=MUTE)
for i in range(3):
    ya = Y0 + H - 8.2 - i * 5.6 - 1.9
    arrow(x2 + (W - 2.4) / 2, ya, x2 + (W - 2.4) / 2, ya - 1.6, color=E[2], lw=0.9, head=5)
_ya = Y0 + H - 8.2 - 2 * 5.6 - 1.9 - 0.8
ax.plot([x2 + (W - 2.4) / 2 + 1.6], [_ya], marker="D", ms=5.0, color=GATE, zorder=6)
text(x2 + (W - 2.4) / 2 + 2.8, _ya, "sev. < 15 dropped", ha="left", fontsize=6.2, color=GATE, style="italic")
yh = Y0 + 3.0
text(X[2] + W / 2, yh, "ranked hypotheses (top 4)", fontsize=7.0, weight="bold", color=E[2])

# Stage 3 contents
x3 = X[3] + 1.2
specs = ["Collision", "Gravity", "Momentum", "Friction", "Deformation", "Fluid", "Causality"]
run = {"Collision", "Gravity", "Fluid"}          # illustrative routed set (top 3)
for i, name in enumerate(specs):
    col, row = i % 2, i // 2
    xx = x3 + col * 11.8
    yy = Y0 + H - 7.4 - row * 3.9
    on = name in run
    box(xx, yy - 1.35, 11.0, 2.7, "white" if on else "#f5f5f5", E[3] if on else "#c4c4c4",
        lw=1.0 if on else 0.6, r=0.5, ls="-" if on else (0, (2, 2)))
    text(xx + 5.5, yy, name, fontsize=7.0, color=INK if on else "#9aa0a6",
         weight="bold" if on else "normal")
text(x3 + 17.3, Y0 + H - 7.4 - 3 * 3.9, "grey = not routed", fontsize=6.0,
     color="#9aa0a6", style="italic")
yv = Y0 + 9.3
box(x3, yv - 2.2, W - 2.4, 4.6, "white", E[3], lw=0.7, r=0.6)
text(x3 + (W - 2.4) / 2, yv + 0.9, "numeric test → VLM verification", fontsize=7.0, weight="bold")
text(x3 + (W - 2.4) / 2, yv - 0.8, "annotated crop of each candidate", fontsize=6.4, color=MUTE)
ax.plot([x3 + 1.0], [yv - 4.0], marker="D", ms=5.0, color=GATE, zorder=6)
text(x3 + 2.2, yv - 4.0, "≤ 3 VLM checks per specialist", ha="left", fontsize=6.2, color=GATE, style="italic")
text(x3 + (W - 2.4) / 2, Y0 + 2.6,
     "findings: confirmed · rejected ·\nunverified", fontsize=6.6, color=E[3], weight="bold")

# Stage 4 contents
x4 = X[4] + 1.2
for i, (name, desc) in enumerate([
        ("Findings timeline", "rejected findings excluded"),
        ("Severity & score", "max severity → level"),
        ("Triage record", "hypotheses kept alongside"),
        ("LLM summary", "text model over the timeline")]):
    yy = Y0 + H - 8.2 - i * 5.3
    box(x4, yy - 1.9, W - 2.4, 4.0, "white", E[4], lw=0.7, r=0.6)
    text(x4 + 1.0, yy + 0.65, name, ha="left", fontsize=7.2, weight="bold")
    text(x4 + 1.0, yy - 0.85, desc, ha="left", fontsize=6.4, color=MUTE)
text(X[4] + W / 2, Y0 + 3.0, "structured report", fontsize=7.4, weight="bold", color=E[4])

# ── inter-stage flow ─────────────────────────────────────────────────────────
yl = Y0 + H - 8.2 - 2 * 5.6          # event localizer row
arrow(X[1] + W, yl, X[2], yl, lw=1.4)
text((X[1] + W + X[2]) / 2, yl + 2.6, "signals", fontsize=6.3, color=MUTE)
yhg = Y0 + H - 8.2 - 3 * 5.6         # hypothesis generator row
arrow(X[2] + W, yhg, X[3], yhg + 5.2, lw=1.4)
ax.plot([(X[2] + W + X[3]) / 2], [yhg + 2.6], marker="D", ms=5.0, color=GATE, zorder=6)
text((X[2] + W + X[3]) / 2, yhg - 0.8, "top 3\nrouted", fontsize=6.2, color=GATE, style="italic")
arrow(X[3] + W, Y0 + 3.0, X[4], Y0 + 3.0, lw=1.4)
text((X[3] + W + X[4]) / 2, Y0 + 5.0, "findings", fontsize=6.3, color=MUTE)
# Stage-1 VLM -> report only, routed along the top edge
ax.plot([X[1] + W - 1.2, X[1] + W + 1.8], [Y0 + 3.4, Y0 + 3.4], color=E[1], lw=0.9,
        ls=(0, (3, 2)), zorder=3)
ax.plot([X[1] + W + 1.8, X[1] + W + 1.8], [Y0 + 3.4, Y0 + H + 1.5], color=E[1], lw=0.9,
        ls=(0, (3, 2)), zorder=3)
ax.plot([X[1] + W + 1.8, X[4] + W / 2], [Y0 + H + 1.5, Y0 + H + 1.5], color=E[1], lw=0.9,
        ls=(0, (3, 2)), zorder=3)
arrow(X[4] + W / 2, Y0 + H + 1.5, X[4] + W / 2, Y0 + H, color=E[1], lw=0.9, ls=(0, (3, 2)), head=5)
text((X[2] + X[3] + W) / 2, Y0 + H + 2.7, "VLM suspicion passed to the report only",
     fontsize=6.4, color=E[1], style="italic")

# ── evidence store (Stages 2–4) ──────────────────────────────────────────────
sx, sw, sy, sh = X[2], X[4] + W - X[2], 1.2, 11.0
box(sx, sy, sw, sh, C["store"], E["store"], lw=1.2, r=1.0)
text(sx + 1.4, sy + sh - 2.0, "Shared evidence store", ha="left", fontsize=8.4, weight="bold",
     color="#3d434b")
text(sx + 1.4, sy + sh - 4.3, "keyed by video content hash · in-process",
     ha="left", fontsize=6.3, color=MUTE)
items = [("tracks, masks", "tracker"), ("kinematics, contacts", "trajectories"),
         ("event windows", "localizer"), ("hypotheses", "triage"),
         ("findings + status", "specialists")]
for i, (a, b) in enumerate(items):
    xx = sx + 1.4 + i * (sw - 2.8) / 5
    box(xx, sy + 1.0, (sw - 2.8) / 5 - 1.0, 3.9, "white", E["store"], lw=0.6, r=0.5)
    text(xx + ((sw - 2.8) / 5 - 1.0) / 2, sy + 3.3, a, fontsize=6.4, weight="bold")
    text(xx + ((sw - 2.8) / 5 - 1.0) / 2, sy + 1.8, b, fontsize=5.8, color=MUTE)
# write / read arrows
arrow(X[2] + W / 2 - 3, Y0, X[2] + W / 2 - 3, sy + sh, color=E[2], lw=1.1)
text(X[2] + W / 2 - 4.2, (Y0 + sy + sh) / 2, "write", fontsize=6.2, color=E[2], ha="right")
arrow(X[3] + W / 2 - 3, Y0, X[3] + W / 2 - 3, sy + sh, color=E[3], lw=1.1)
text(X[3] + W / 2 - 4.2, (Y0 + sy + sh) / 2, "write", fontsize=6.2, color=E[3], ha="right")
arrow(X[3] + W / 2 + 3, sy + sh, X[3] + W / 2 + 3, Y0, color=E[3], lw=1.1, ls=(0, (3, 2)))
text(X[3] + W / 2 + 4.2, (Y0 + sy + sh) / 2, "read", fontsize=6.2, color=E[3], ha="left")
arrow(X[4] + W / 2, sy + sh, X[4] + W / 2, Y0, color=E[4], lw=1.1, ls=(0, (3, 2)))
text(X[4] + W / 2 + 1.2, (Y0 + sy + sh) / 2, "read", fontsize=6.2, color=E[4], ha="left")
text(X[1] + W / 2, sy + sh / 2, "Stage 1 is stateless:\nno reads or writes",
     fontsize=6.4, color=MUTE, style="italic")

# legend for gates
ax.plot([X[1] + 1.4], [sy + 1.6], marker="D", ms=5, color=GATE)
text(X[1] + 2.6, sy + 1.6, "gate implemented in code", ha="left", fontsize=6.2, color=GATE)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"pipeline.{ext}", bbox_inches="tight", dpi=220)
print("wrote", OUT / "pipeline.pdf")
