"""
Figure replacing the physics-family table (fig:category-accuracy).

Left, detection (this family vs. clean videos): an arrow per family from the
best single VLM to the best VLM + PhysicsLENS Stage-1/2 signals, labelled with
the change. Right, attribution (this family vs. a different violation): a
lollipop from chance to the mean of ten VLMs (+/- sd), with the best VLM and
signals-only as secondary markers. Values are those of the former table.

python paper/fig/make_category_plot.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
# family, n, attribution: best VLM, mean, sd, signals-only | detection: best VLM, + signals
ROWS = [
    ("Fluid",       35, 0.87, 0.82, 0.03, 0.53, 0.84, 0.80),
    ("Gravity",     44, 0.66, 0.55, 0.08, 0.56, 0.72, 0.75),
    ("Deformation", 146, 0.61, 0.58, 0.03, 0.60, 0.68, 0.74),
    ("Collision",   167, 0.57, 0.54, 0.02, 0.45, 0.67, 0.73),
    ("Friction",    24, 0.65, 0.57, 0.05, 0.48, 0.66, 0.69),
    ("Causality",   169, 0.59, 0.54, 0.03, 0.53, 0.56, 0.69),
    ("Momentum",    59, 0.58, 0.53, 0.04, 0.46, 0.67, 0.67),
]
BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#6b6a66", "#ecebe7"

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True, gridspec_kw={"wspace": 0.08})
y = list(range(len(ROWS)))[::-1]

for ax in (a1, a2):
    ax.axvspan(0.40, 0.5, color="#f4f3f0", zorder=0, lw=0)
    ax.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(0.40, 0.92)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=MUTED, length=2.5, pad=2)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)

# ── detection: best VLM -> + PhysicsLENS signals ──
for yi, r in zip(y, ROWS):
    b, s = r[6], r[7]
    col = BLUE if s >= b else ORANGE
    a1.plot([b, s], [yi, yi], color=col, lw=2.2, alpha=0.55, zorder=2, solid_capstyle="round")
    a1.plot(b, yi, "o", ms=5.5, mfc="white", mec=GREY, mew=1.3, zorder=3)
    a1.plot(s, yi, "o", ms=5.5, color=col, mec="white", mew=0.7, zorder=4)
    d = s - b
    a1.text(max(s, b) + 0.018, yi, f"{d:+.2f}", va="center", ha="left", fontsize=7.5,
            color=col if abs(d) > 0.005 else MUTED)
a1.set_title("Detection  (vs. clean videos)", fontsize=8.5, color=INK, pad=3, loc="left")

# ── attribution: chance -> mean of 10 VLMs ──
for yi, r in zip(y, ROWS):
    best, m, sd, sig = r[2], r[3], r[4], r[5]
    strong = m > 0.7
    col = BLUE if strong else "#7aa9e3"
    a2.plot([0.5, m], [yi, yi], color=col, lw=1.4, zorder=2, solid_capstyle="round")
    a2.plot([m - sd, m + sd], [yi, yi], color=col, lw=3.2, alpha=0.35, zorder=2, solid_capstyle="round")
    a2.plot(m, yi, "o", ms=5.5, color=col, mec="white", mew=0.7, zorder=4)
    a2.plot(best, yi, "o", ms=5.5, mfc="white", mec=GREY, mew=1.3, zorder=3)
    a2.plot(sig, yi, "s", ms=3.6, color=GREY, mec="white", mew=0.5, zorder=3)
a2.set_title("Attribution  (vs. other violations)", fontsize=8.5, color=INK, pad=3, loc="left")

a1.set_yticks(y)
a1.set_yticklabels([r[0] for r in ROWS])
for t, r in zip(a1.get_yticklabels(), ROWS):
    t.set_color(INK)
for yi, r in zip(y, ROWS):
    a1.text(0.405, yi, f"n={r[1]}", va="center", ha="left", fontsize=6.5, color=MUTED)
for ax in (a1, a2):
    ax.set_xlabel("AUC", color=MUTED, labelpad=1, fontsize=8)

h = [plt.Line2D([], [], marker="o", ls="", ms=5.5, mfc="white", mec=GREY, mew=1.3, label="Best VLM"),
     plt.Line2D([], [], marker="o", ls="", ms=5.5, color=BLUE, mec="white", label="Best VLM + PhysicsLENS signals"),
     plt.Line2D([], [], marker="o", ls="-", lw=3.2, ms=5.5, color="#7aa9e3", alpha=0.8, mec="white", label="Mean of 10 VLMs $\\pm$ sd"),
     plt.Line2D([], [], marker="s", ls="", ms=3.6, color=GREY, label="Signals only")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.56, 0.0), ncol=4, frameon=False,
           fontsize=7.5, handletextpad=0.3, columnspacing=1.1, handlelength=1.5)
fig.subplots_adjust(left=0.135, right=0.99, top=0.91, bottom=0.25)
fig.savefig(OUT / "fig_category_accuracy.pdf")
fig.savefig(OUT / "fig_category_accuracy.png", dpi=220)
print("->", OUT / "fig_category_accuracy.pdf")
