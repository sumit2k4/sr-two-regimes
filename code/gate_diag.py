"""Diagnose the headroom available to any per-image gate.

If an oracle that picks the best branch for every image is barely better than
uniform averaging, then the branches make the same mistakes and no gating rule
of any kind can help. If the oracle is far above, the headroom exists and the
failure is one of gate estimation rather than of the idea.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from qge import load_branches, load_arch_ensemble, metrics


def report(ds, protocol="p1"):
    sp = C.get_splits(ds)
    ncls = len(sp["classes"])
    rows = {}
    for tag, loader in [("views", "v"), ("extended", "e")]:
        accs = {}
        for seed in C.CLS_SEEDS:
            Pv, Pt, yv, yt = load_branches(ds, seed, protocol)
            if loader == "e":
                Av, At = load_arch_ensemble(ds, seed, protocol)
                if At is None:
                    continue
                Pt = np.concatenate([Pt, At[:, 1:]], 1)      # add EffB0, DN121
            K = Pt.shape[1]
            correct = np.stack([Pt[:, k].argmax(1) == yt for k in range(K)], 1)
            oracle = correct.any(1).mean()
            allright = correct.all(1).mean()
            nobody = (~correct.any(1)).mean()
            avg = (Pt.mean(1).argmax(1) == yt).mean()
            best_single = correct.mean(0).max()
            # accuracy of the best fixed branch chosen per class (a weaker oracle)
            accs.setdefault("oracle_per_image", []).append(float(oracle))
            accs.setdefault("avg", []).append(float(avg))
            accs.setdefault("best_single", []).append(float(best_single))
            accs.setdefault("all_branches_correct", []).append(float(allright))
            accs.setdefault("no_branch_correct", []).append(float(nobody))
            accs.setdefault("K", []).append(K)
        if accs:
            rows[tag] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in accs.items()}
    return rows


if __name__ == "__main__":
    out = {}
    for ds in ["ucmerced", "eurosat"]:
        for p in ["p1", "p2"]:
            try:
                out[ds + "_" + p] = report(ds, p)
            except Exception as e:
                print("skip", ds, p, e)
    json.dump(out, open(os.path.join(C.RESULTS, "gate_headroom.json"), "w"), indent=1)
    for k, v in out.items():
        print("=" * 60)
        print(k)
        for tag, d in v.items():
            print("  %-9s K=%d  best-single %.4f  average %.4f  ORACLE %.4f  "
                  "| all-correct %.4f  none-correct %.4f"
                  % (tag, d["K"][0], d["best_single"][0], d["avg"][0],
                     d["oracle_per_image"][0], d["all_branches_correct"][0],
                     d["no_branch_correct"][0]))
