"""Figure 5: paired plausibility change (unobservable - observable) per model and
pooled, with 95% bootstrap intervals over scenarios (values as reported in
Section 5.2). Drawn at the size of the half-column beside Table 5.

python paper/fig/make_forest_plausibility.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
W_PT, H_PT = (float(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (193, 100)
ROWS = [  # label, delta, lo, hi, colour  (same order as Table 5)
    ("HunyuanVideo 1.5", -0.38, -0.93, 0.17, "#7b5bbf"),
    ("MAGI 4.5B distill", -0.23, -0.73, 0.27, "#c0463f"),
    ("Wan 2.2", -0.30, -0.63, 0.00, "#3a6ea5"),
    ("Cosmos-nano 1", -0.13, -0.53, 0.27, "#cc8f2a"),
    ("Pooled", -0.276, -0.552, 0.017, "black"),
]
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(W_PT / 72, H_PT / 72))
n = len(ROWS)
for i, (lab, d, lo, hi, c) in enumerate(ROWS):
    y = n - 1 - i + (0 if lab != "Pooled" else -0.25)
    pooled = lab == "Pooled"
    ax.plot([lo, hi], [y, y], color=c, lw=1.8 if pooled else 1.2, solid_capstyle="butt")
    for x in (lo, hi):
        ax.plot([x, x], [y - 0.18, y + 0.18], color=c, lw=1.8 if pooled else 1.2)
    ax.plot(d, y, marker="D" if pooled else "o", ms=5 if pooled else 4.2,
            color=c, mec="white", mew=0.5, zorder=3)
ax.axvline(0, color="#777", lw=0.7, ls="--")
ax.axhline(0.45, color="#bbb", lw=0.5)
ax.set_yticks([n - 1 - i + (0 if r[0] != "Pooled" else -0.25) for i, r in enumerate(ROWS)])
ax.set_yticklabels([r[0] for r in ROWS])
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xlim(-1.0, 0.35)
ax.set_xticks([-0.75, -0.5, -0.25, 0, 0.25])
ax.set_xticklabels(["$-0.75$", "$-0.5$", "$-0.25$", "0", "0.25"])
ax.set_ylim(-0.75, n - 0.5)
ax.tick_params(axis="y", length=0, pad=2)
ax.tick_params(axis="x", length=2.5, pad=1.5)
ax.set_xlabel(r"$\Delta = P^{\mathrm{unobs}} - P^{\mathrm{obs}}$", labelpad=1)
ax.grid(axis="x", color="#e6e6e6", lw=0.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.15)
fig.savefig(OUT / "fig_forest_plausibility.pdf")
fig.savefig(OUT / "fig_forest_plausibility.png", dpi=300)
print("->", OUT / "fig_forest_plausibility.pdf", W_PT, H_PT)
