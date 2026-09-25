"""
Figure replacing the physics-family table (tab:category-accuracy): per family,
attribution AUC (this family vs. a different violation) and detection AUC (this
family vs. clean videos). Values are those of paper_tables.py's categories
table. Marker shape + colour both encode the series (colour-blind safe palette,
validated); a dashed line marks chance.

python paper/fig/make_category_plot.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
# family, n_pos, attribution: best VLM, mean, sd, signals | detection: best VLM, best VLM + signals
ROWS = [
    ("Fluid",       35, 0.87, 0.82, 0.03, 0.53, 0.84, 0.80),
    ("Gravity",     44, 0.66, 0.55, 0.08, 0.56, 0.72, 0.75),
    ("Friction",    24, 0.65, 0.57, 0.05, 0.48, 0.66, 0.69),
    ("Deformation", 146, 0.61, 0.58, 0.03, 0.60, 0.68, 0.74),
    ("Causality",   169, 0.59, 0.54, 0.03, 0.53, 0.56, 0.69),
    ("Momentum",    59, 0.58, 0.53, 0.04, 0.46, 0.67, 0.67),
    ("Collision",   167, 0.57, 0.54, 0.02, 0.45, 0.67, 0.73),
]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.45), sharey=True,
                             gridspec_kw={"wspace": 0.06})
y = list(range(len(ROWS)))[::-1]
labels = [f"{r[0]} ({r[1]})" for r in ROWS]
ring = dict(mec="white", mew=0.8, zorder=3)
for yi, r in zip(y, ROWS):
    _, _, ab, am, asd, asig, db, dsig = r
    a1.errorbar(am, yi, xerr=asd, fmt="o", color=BLUE, ms=5.5, elinewidth=1.2, capsize=0, **ring)
    a1.plot(ab, yi, "D", color=ORANGE, ms=5, **ring)
    a1.plot(asig, yi, "s", color=AQUA, ms=5, **ring)
    a2.plot(db, yi, "D", color=ORANGE, ms=5, **ring)
    a2.plot(dsig, yi, "^", color=YELLOW, ms=6, **ring)
for ax, t in ((a1, "Attribution: this family vs. another violation"),
              (a2, "Detection: this family vs. clean videos")):
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(0.42, 0.92)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_title(t, fontsize=8.5, color=INK, pad=4)
    ax.set_xlabel("AUC", color=MUTED, labelpad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, pad=2)
a1.set_yticks(y)
a1.set_yticklabels(labels)
a2.tick_params(axis="y", length=0)
h = [plt.Line2D([], [], marker="o", color=BLUE, ls="-", lw=1.2, ms=5.5, mec="white", label="Mean of 10 VLMs (± sd)"),
     plt.Line2D([], [], marker="D", color=ORANGE, ls="", ms=5, mec="white", label="Best VLM"),
     plt.Line2D([], [], marker="s", color=AQUA, ls="", ms=5, mec="white", label="Signals only"),
     plt.Line2D([], [], marker="^", color=YELLOW, ls="", ms=6, mec="white", label="Best VLM + signals")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8,
           bbox_to_anchor=(0.55, -0.02), handletextpad=0.3, columnspacing=1.2)
fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.27)
fig.savefig(OUT / "fig_category_accuracy.pdf")
fig.savefig(OUT / "fig_category_accuracy.png", dpi=220)
print("->", OUT / "fig_category_accuracy.pdf")
