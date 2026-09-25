"""Figure 4: mean plausibility P vs. mean hidden-property adherence H, one point
per model (values from Table 4). Drawn at its printed size (the right-hand
minipage next to Table 4) so fonts are true point sizes.

python paper/fig/make_mean_dotplot.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
# model: (mean H, mean P, colour, label offset in points, alignment)
MODELS = {
    "MAGI 4.5B distill": (1.10, 3.10, "#c0463f", (7, -2), "left"),
    "HunyuanVideo 1.5":  (2.24, 2.24, "#7b5bbf", (-6, 2), "right"),
    "Cosmos-nano 1":     (1.87, 2.07, "#cc8f2a", (-6, -6), "right"),
    "Wan 2.2":           (1.90, 1.53, "#3a6ea5", (7, -2), "left"),
}

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9})
fig, ax = plt.subplots(figsize=(179 / 72, 65 / 72))   # the space beside Table 4, in pt
for name, (h, p, c, off, ha) in MODELS.items():
    ax.scatter(h, p, s=70, color=c, zorder=3, edgecolor="white", linewidth=0.6)
    ax.annotate(name, (h, p), xytext=off, textcoords="offset points",
                ha=ha, va="center", fontsize=8.5)
ax.set_xlim(0.9, 2.45)
ax.set_ylim(1.2, 3.4)
ax.set_xticks([1.0, 1.5, 2.0, 2.5])
ax.set_yticks([1, 2, 3])
ax.tick_params(labelsize=8.5, length=2.5, pad=1.5)
ax.set_xlabel(r"Adherence $H$", fontsize=9, labelpad=1)
ax.set_ylabel(r"Plaus. $P$", fontsize=9, labelpad=1)
ax.grid(color="#e6e6e6", linewidth=0.5, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.15)
fig.savefig(OUT / "fig_mean_dotplot.pdf")
fig.savefig(OUT / "fig_mean_dotplot.png", dpi=300)
print("->", OUT / "fig_mean_dotplot.pdf")
