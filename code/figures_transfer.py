"""Figures for the transfer regime and the gate-headroom analysis."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from figures import save, COL1, COL2, CB, SR_MODELS

PRETTY = {"bic": "Bicubic", "srcnn": "SRCNN", "edsr": "EDSR",
          "hfgan_p": "HFGAN-P", "hfgan_g": "HFGAN-G"}
TR = json.load(open(os.path.join(C.RESULTS, "transfer.json")))
T = json.load(open(os.path.join(C.RESULTS, "all_tables.json")))


def fig_two_regimes():
    """The headline figure: the same reconstructions, measured in both regimes."""
    dss = [d for d in ["ucmerced", "eurosat"] if d in TR]
    fig, axes = plt.subplots(1, len(dss), figsize=(COL2, 2.5), squeeze=False)
    for j, ds in enumerate(dss):
        ax = axes[0][j]
        b = TR[ds]["transfer"]
        cf = T[ds]["confound"]
        ms = [m for m in SR_MODELS if m in b and m in cf["models"]]
        x = np.arange(len(ms))
        w = 0.38
        matched = [cf["models"][m]["controlled_gain"] * 100 for m in ms]
        transfer = [b[m]["gain"] * 100 for m in ms]
        ax.bar(x - w / 2, matched, w, color=CB[0],
               label="matched: classifier retrained on the reconstruction")
        ax.bar(x + w / 2, transfer, w, color=CB[3],
               label="transfer: HR-trained classifier reused")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[m] for m in ms], rotation=12)
        ax.set_ylabel("OA gain over the bicubic control (pp)" if j == 0 else "")
        ax.set_title("UC Merced" if ds == "ucmerced" else "EuroSAT")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(top=max(max(transfer), max(matched)) * 1.28)
        for xi, (a, c) in enumerate(zip(matched, transfer)):
            ax.text(xi - w / 2, a, "%+.1f" % a, ha="center",
                    va="bottom" if a >= 0 else "top", fontsize=6)
            ax.text(xi + w / 2, c, "%+.1f" % c, ha="center", va="bottom", fontsize=6)
    axes[0][0].legend(loc="upper left", frameon=False, fontsize=6.2)
    save(fig, "fig_two_regimes")


def fig_lpips_transfer():
    """Perceptual distance against transfer gain, per class, pooled over models."""
    dss = [d for d in ["ucmerced", "eurosat"] if d in TR]
    fig, axes = plt.subplots(1, 2 * len(dss), figsize=(COL2, 1.95), squeeze=False)
    for j, ds in enumerate(dss):
        for c, (src, key, lab) in enumerate(
                [(T[ds]["quality_vs_gain"], "d_lpips", "matched"),
                 (TR[ds]["transfer_corr"], "d_lpips", "transfer")]):
            ax = axes[0][2 * j + c]
            pts = src["points"]
            for i, m in enumerate(SR_MODELS):
                sel = [p for p in pts if p["model"] == m]
                if not sel:
                    continue
                ax.scatter([p["d_lpips"] for p in sel],
                           [p["d_acc"] * 100 for p in sel], s=9, color=CB[i],
                           label=PRETTY[m] if (j == 0 and c == 0) else None,
                           alpha=0.8, linewidth=0)
            x = np.array([p["d_lpips"] for p in pts])
            y = np.array([p["d_acc"] for p in pts]) * 100
            if len(x) > 2:
                bb = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 20)
                ax.plot(xs, np.polyval(bb, xs), color="k", lw=0.8, ls="--")
            ax.axhline(0, color="grey", lw=0.5)
            v = src[key]
            ax.set_title("%s, %s\n$\\rho$=%+.2f (p=%.1g)"
                         % ("UCM" if ds == "ucmerced" else "EuroSAT", lab,
                            v["spearman"], v["spearman_p"]), fontsize=6.6)
            ax.set_xlabel(r"$\Delta$LPIPS", fontsize=7)
            if c == 0 and j == 0:
                ax.set_ylabel(r"$\Delta$OA (pp)")
            ax.grid(alpha=0.3)
    axes[0][0].legend(frameon=False, loc="best", fontsize=5.5, ncol=2)
    save(fig, "fig_lpips_transfer")


def fig_headroom():
    f = os.path.join(C.RESULTS, "gate_headroom.json")
    if not os.path.exists(f):
        return
    H = json.load(open(f))
    keys = [k for k in ["ucmerced_p1", "eurosat_p1"] if k in H]
    fig, axes = plt.subplots(1, len(keys), figsize=(COL2, 2.2), squeeze=False)
    for j, k in enumerate(keys):
        ax = axes[0][j]
        tags = [t for t in ["views", "extended"] if t in H[k]]
        x = np.arange(len(tags))
        w = 0.26
        bs = [H[k][t]["best_single"][0] * 100 for t in tags]
        av = [H[k][t]["avg"][0] * 100 for t in tags]
        orc = [H[k][t]["oracle_per_image"][0] * 100 for t in tags]
        ax.bar(x - w, bs, w, color=CB[0], label="best single branch")
        ax.bar(x, av, w, color=CB[2], label="posterior average")
        ax.bar(x + w, orc, w, color=CB[3], label="per-image oracle")
        for xi in range(len(tags)):
            ax.annotate("", xy=(xi + w, orc[xi]), xytext=(xi, av[xi]),
                        arrowprops=dict(arrowstyle="<->", lw=0.6, color="k"))
            ax.text(xi + w / 2, (orc[xi] + av[xi]) / 2, " %.2f pp" % (orc[xi] - av[xi]),
                    fontsize=6, va="center")
        ax.set_xticks(x)
        ax.set_xticklabels(["3 views" if t == "views" else "5 experts" for t in tags])
        ax.set_ylim(min(bs) - 1.5, max(orc) + 0.8)
        ax.set_ylabel("OA (%)" if j == 0 else "")
        ax.set_title("UC Merced" if k.startswith("uc") else "EuroSAT")
        ax.grid(axis="y", alpha=0.3)
        if j == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower right")
    save(fig, "fig_headroom")


if __name__ == "__main__":
    for f in [fig_two_regimes, fig_lpips_transfer, fig_headroom]:
        try:
            f()
        except Exception as e:
            print("  SKIP", f.__name__, type(e).__name__, e)
