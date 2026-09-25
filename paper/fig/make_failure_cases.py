"""Figure: three annotated failure cases (Section 5.3). Frames are sampled from
the generated videos; ratings and issue text are the annotators'.

python paper/fig/make_failure_cases.py
"""
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CASES = [  # clip, frame indices, row label
    ("wan__unobs__019", [17, 51, 85, 120],
     r"(a) Wan 2.2: tea stated to be molasses-thick pours like water ($P{=}4$, $H{=}1$)"),
    ("hunyuan__unobs__022", [17, 51, 85, 120],
     r"(b) HunyuanVideo 1.5: rated as following the liquid-repellent surface, but the glass moves untouched ($P{=}1$, $H{=}3$)"),
    ("magi__unobs__011", [0, 40, 80, 119],
     r"(c) MAGI 4.5B: the robot barely moves; rated fully plausible ($P{=}4$, $H{=}1$, not completed)"),
]

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
fig, axes = plt.subplots(len(CASES), 4, figsize=(5.5, 3.55),
                         gridspec_kw={"hspace": 0.32, "wspace": 0.03})
for r, (clip, idx, label) in enumerate(CASES):
    cap = cv2.VideoCapture(str(ROOT / "data/consolidated_videos" / f"{clip}.mp4"))
    for c, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        ax = axes[r, c]
        ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        ax.text(0.03, 0.95, f"t={i / 24:.1f}s", transform=ax.transAxes, fontsize=6,
                va="top", color="white",
                bbox=dict(facecolor="black", alpha=0.5, pad=1, linewidth=0))
    axes[r, 0].set_title(label, loc="left", fontsize=8, pad=2.5,
                         x=0, transform=axes[r, 0].transAxes)
fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005)
fig.savefig(OUT / "fig_failure_cases.pdf", dpi=220)
print("->", OUT / "fig_failure_cases.pdf")
