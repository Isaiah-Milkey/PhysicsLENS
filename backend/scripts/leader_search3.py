"""
Round 3: condition-aware plausibility for unobservable videos. The evaluator
always knows whether the property was stated in text (it reads the prompt), so
fitting the combination separately on unobservable training videos is allowed.
Same held-out protocol: weights / selection / regularisation on training folds.

  un-fuse     ten-VLM ensemble (debiased) + signals, weight fitted on
              unobservable training videos only
  un-fuseT    same with temporal-embedding features in the signal pool
  un-stack    L2-logistic on ensemble answers (debiased, plain, hidden,
              hidden-signature, completion) + signals, unobservable only
  un-topK     top-5 backbones chosen on unobservable training videos
  un-topK+sig that plus signals (fitted weight, unobservable only)

python backend/scripts/leader_search3.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import leader_search as L  # noqa: E402
import leader_search2 as L2  # noqa: E402  (reuses its helpers; its own run is cached below)
from fusion_utils import make_fused  # noqa: E402
from specialist_accuracy import auc  # noqa: E402
from system_eval import rank01  # noqa: E402

UN, YP, N = L.UN, L.YP, L.N
fz28, fzT = make_fused(L.X, L.G, L.F, N), make_fused(L.XS, L.G, L.F, N)
E = L.ENS
yi = YP.astype(int)


def auc_un(s, idx=None):
    idx = np.arange(N) if idx is None else idx
    m = UN[idx] & np.isfinite(s[idx])
    return auc(yi[idx][m], s[idx][m])


cands = {
    "un-fuse": fz28(E["deb"], L.Y, yi, UN),
    "un-fuseT": fzT(E["deb"], L.Y, yi, UN),
    "un-stack": L.stack_oof(np.hstack([np.column_stack([rank01(E[k]) for k in
                                                       ["deb", "plaus", "hid", "hsig", "act"]]), L.XS]),
                            YP, UN),
    "un-topK": L2.topk("deb", YP, UN),
}
v = cands["un-topK"]
cands["un-topK+sig"] = fz28(np.nan_to_num(v, nan=np.nanmean(v)), L.Y, yi, UN)
out = {}
for k, s in cands.items():
    b = [auc_un(s, i) for i in L.BOOT]
    out[k] = (auc_un(s), float(np.nanstd(b)))
    print(f"{k:14s} unobs AUC {out[k][0]:.3f} ± {out[k][1]:.3f}")
json.dump(out, open(L.P.D / "leader_search3.json", "w"), indent=1)
