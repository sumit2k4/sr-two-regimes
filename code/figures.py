"""Generate every figure of the manuscript (IEEE column geometry, 600 dpi)."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
    "savefig.dpi": 600, "figure.dpi": 120,
})
COL1, COL2 = 3.5, 7.16
PRETTY = {"bic": "Bicubic", "srcnn": "SRCNN", "edsr": "EDSR",
          "hfgan_p": "HFGAN-P", "hfgan_g": "HFGAN-G", "hr": "HR", "lr": "LR"}
SR_MODELS = ["srcnn", "edsr", "hfgan_p", "hfgan_g"]
CB = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def save(fig, name):
    p = os.path.join(C.FIGS, name)
    fig.tight_layout(pad=0.25)
    fig.savefig(p + ".png", bbox_inches="tight")
    fig.savefig(p + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


T = json.load(open(os.path.join(C.RESULTS, "all_tables.json")))


# ---------------------------------------------------------------- Fig. 1
def fig_framework():
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")
    EC = "#333333"

    def box(x0, x1, y0, y1, t, fc, fs=7.2):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.35", fc=fc, ec=EC, lw=0.7))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, t, ha="center", va="center", fontsize=fs)

    def arr(p0, p1, ls="-", col=EC):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                     lw=0.7, ls=ls, color=col, shrinkA=0, shrinkB=0))

    ys = [50.5, 36.5, 22.5]
    box(2, 14, 30.5, 42.5, "LR scene\n$x_{\\mathrm{LR}}$", "#EDEDED")
    names = ["Bicubic $\\uparrow$4\n(geometry faithful)",
             "EDSR\n(distortion oriented)",
             "HFGAN-G [Paper 2]\n(perception oriented)"]
    for y, nm in zip(ys, names):
        box(18, 33, y - 5, y + 5, nm, "#DCE6F1", fs=6.8)
        arr((14.4, 36.5), (17.6, y))
    for i, y in enumerate(ys):
        box(38, 50, y - 5, y + 5, "CNN $f_%d$\n$\\to p_%d$" % (i + 1, i + 1), "#E8F0DC")
        arr((33.4, y), (37.6, y))
    box(56, 71, 3, 15, "reliability descriptor\n$z(x)$ : 20-D,\nno ground truth", "#FBE5D6", fs=6.8)
    box(76, 92, 3, 15, "gate $g_{\\phi}$\n$w \\in \\Delta^{2}$", "#FBE5D6")
    box(72, 98, 30.5, 44.5, "$p=\\sum_k w_k\\,p_k$\nland-cover label", "#F2DCDB", fs=7.6)

    for y in ys:
        ax.plot([33.4, 35.2], [y - 3.0, y - 3.0], ls=":", lw=0.7, color="#666666")
        ax.plot([35.2, 35.2], [y - 3.0, 6.5], ls=":", lw=0.7, color="#666666")
    ax.plot([35.2, 54.5], [6.5, 6.5], ls=":", lw=0.7, color="#666666")
    arr((54.5, 6.5), (55.6, 6.5), ls=":", col="#666666")
    for y in ys:
        ax.plot([50.4, 52.6], [y - 3.0, y - 3.0], ls=":", lw=0.7, color="#666666")
        ax.plot([52.6, 52.6], [y - 3.0, 11.5], ls=":", lw=0.7, color="#666666")
    ax.plot([52.6, 54.5], [11.5, 11.5], ls=":", lw=0.7, color="#666666")
    arr((54.5, 11.5), (55.6, 11.5), ls=":", col="#666666")
    arr((71.4, 9), (75.6, 9))
    arr((84, 15.4), (84, 30.1))
    for y in ys:
        arr((50.4, y + 2.0), (71.6, 37.5))
    ax.text(45.0, 3.2, "no-reference image statistics", fontsize=6, color="#666666", ha="center")
    ax.text(51.8, 13.6, "branch posteriors", fontsize=6, color="#666666", ha="right")
    ax.text(25.5, 57.0, "(a) one scene, three points of the\nperception-distortion trade-off",
            fontsize=6.6, ha="center", style="italic")
    ax.text(84, 18.5, "(b) per-image mixing weights", fontsize=6.6,
            ha="center", style="italic")
    save(fig, "fig1_framework")


# ---------------------------------------------------------------- Fig. 2
def fig_confound():
    dss = [d for d in ["ucmerced", "eurosat"] if d in T]
    fig, axes = plt.subplots(1, len(dss), figsize=(COL2, 2.1), squeeze=False)
    for j, ds in enumerate(dss):
        ax = axes[0][j]
        cf = T[ds]["confound"]
        ms = [m for m in SR_MODELS if m in cf["models"]]
        x = np.arange(len(ms))
        w = 0.38
        naive = [cf["models"][m]["naive_gain"] * 100 for m in ms]
        ctrl = [cf["models"][m]["controlled_gain"] * 100 for m in ms]
        ax.bar(x - w / 2, naive, w, color=CB[1], label="vs LR-native (uncontrolled)")
        ax.bar(x + w / 2, ctrl, w, color=CB[0], label="vs bicubic control")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[m] for m in ms], rotation=12)
        ax.set_ylabel("OA gain (pp)" if j == 0 else "")
        ax.set_title("UC Merced" if ds == "ucmerced" else "EuroSAT")
        ax.grid(axis="y", alpha=0.3)
        for xi, (a, b) in enumerate(zip(naive, ctrl)):
            ax.text(xi - w / 2, a, "%+.1f" % a, ha="center",
                    va="bottom" if a >= 0 else "top", fontsize=6)
            ax.text(xi + w / 2, b, "%+.1f" % b, ha="center",
                    va="bottom" if b >= 0 else "top", fontsize=6)
    axes[0][0].legend(loc="best", frameon=False)
    save(fig, "fig2_confound")


# ---------------------------------------------------------------- Fig. 3
def fig_pd_plane():
    dss = [d for d in ["ucmerced", "eurosat"] if d in T and T[d].get("sr_quality")]
    fig, axes = plt.subplots(1, len(dss), figsize=(COL2, 2.3), squeeze=False)
    for j, ds in enumerate(dss):
        ax = axes[0][j]
        q = T[ds]["sr_quality"]
        cf = T[ds]["confound"]
        for i, m in enumerate(["bic"] + SR_MODELS):
            k = m + "_p1"
            if k not in q:
                continue
            acc = cf["bic"][0] if m == "bic" else cf["models"][m]["sr"][0]
            ax.scatter(q[k]["psnr"][0], q[k]["lpips"][0],
                       s=20 + 700 * max(0.0, acc - 0.55),
                       color=CB[i], zorder=3, edgecolor="k", linewidth=0.4)
            ax.annotate("%s\n%.1f%%" % (PRETTY[m], acc * 100),
                        (q[k]["psnr"][0], q[k]["lpips"][0]),
                        textcoords="offset points", xytext=(6, 4), fontsize=6.2)
        ax.set_xlabel("PSNR (dB)")
        ax.set_ylabel("LPIPS (lower is better)" if j == 0 else "")
        ax.set_title("UC Merced" if ds == "ucmerced" else "EuroSAT")
        ax.margins(0.22)
        ax.grid(alpha=0.3)
    save(fig, "fig3_perception_distortion")


# ---------------------------------------------------------------- Fig. 4
def fig_perclass(ds="ucmerced"):
    d = T.get(ds, {}).get("perclass_hfgan_g")
    if not d:
        return
    o = np.array(d["order"])
    delta = np.array(d["delta"])[o] * 100
    names = [d["classes"][i] for i in o]
    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    cols = [CB[2] if v > 0 else (CB[3] if v < 0 else "#999999") for v in delta]
    ax.bar(np.arange(len(delta)), delta, color=cols, edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(len(delta)))
    ax.set_xticklabels(names, rotation=62, ha="right", fontsize=6)
    ax.set_ylabel("OA gain over\nbicubic control (pp)", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig4_perclass_" + ds)


# ---------------------------------------------------------------- Fig. 5
def fig_corr():
    dss = [d for d in ["ucmerced", "eurosat"] if d in T and "quality_vs_gain" in T[d]]
    if not dss:
        return
    fig, axes = plt.subplots(len(dss), 3, figsize=(COL2, 2.0 * len(dss)), squeeze=False)
    for r, ds in enumerate(dss):
        qv = T[ds]["quality_vs_gain"]
        pts = qv["points"]
        for c, key in enumerate(["d_psnr", "d_ssim", "d_lpips"]):
            ax = axes[r][c]
            for i, m in enumerate(SR_MODELS):
                sel = [p for p in pts if p["model"] == m]
                if not sel:
                    continue
                ax.scatter([p[key] for p in sel], [p["d_acc"] * 100 for p in sel],
                           s=9, color=CB[i],
                           label=PRETTY[m] if (r == 0 and c == 0) else None,
                           alpha=0.8, linewidth=0)
            x = np.array([p[key] for p in pts])
            y = np.array([p["d_acc"] for p in pts]) * 100
            if len(x) > 2:
                b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 20)
                ax.plot(xs, np.polyval(b, xs), color="k", lw=0.8, ls="--")
            ax.axhline(0, color="grey", lw=0.5)
            v = qv[key]
            ax.set_title(r"$\rho$=%+.2f  (p=%.1e)" % (v["spearman"], v["spearman_p"]),
                         fontsize=7)
            ax.set_xlabel({"d_psnr": r"$\Delta$PSNR (dB)", "d_ssim": r"$\Delta$SSIM",
                           "d_lpips": r"$\Delta$LPIPS"}[key])
            if c == 0:
                ax.set_ylabel(("UC Merced" if ds == "ucmerced" else "EuroSAT")
                              + "\n" + r"$\Delta$OA (pp)")
            ax.grid(alpha=0.3)
    axes[0][0].legend(frameon=False, loc="best", ncol=2, fontsize=6)
    save(fig, "fig5_quality_vs_gain")


# ---------------------------------------------------------------- Fig. 6
def fig_gate():
    dss = [d for d in ["ucmerced", "eurosat"]
           if os.path.exists(os.path.join(C.RESULTS, "qgeweights_" + d + "_p1_42.npz"))]
    if not dss:
        return
    fig, axes = plt.subplots(1, 2 * len(dss), figsize=(COL2, 2.1), squeeze=False)
    for j, ds in enumerate(dss):
        z = np.load(os.path.join(C.RESULTS, "qgeweights_" + ds + "_p1_42.npz"))
        w, y = z["w"], z["y"]
        ax = axes[0][2 * j]
        for i, lab in enumerate(["bicubic", "EDSR", "HFGAN-G"]):
            ax.hist(w[:, i], bins=24, range=(0, 1), histtype="step",
                    color=CB[i], label=lab, lw=1.0)
        ax.set_xlabel("gate weight $w_k$")
        ax.set_ylabel("test images" if j == 0 else "")
        ax.set_title("UC Merced" if ds == "ucmerced" else "EuroSAT", fontsize=7.5)
        ax.legend(frameon=False, fontsize=6)
        ax.grid(alpha=0.3)
        ax = axes[0][2 * j + 1]
        wg = w[:, 2]
        qs = np.quantile(wg, [0, .25, .5, .75, 1.0])
        aq, ab, ag = [], [], []
        for i in range(4):
            m = (wg >= qs[i]) & (wg <= qs[i + 1])
            if m.sum() == 0:
                aq.append(np.nan); ab.append(np.nan); ag.append(np.nan); continue
            aq.append((z["pred_qge"][m] == y[m]).mean() * 100)
            ab.append((z["pred_bic"][m] == y[m]).mean() * 100)
            ag.append((z["pred_gan"][m] == y[m]).mean() * 100)
        x = np.arange(4)
        ax.plot(x, ab, "o-", color=CB[0], ms=3, label="bicubic branch")
        ax.plot(x, ag, "s-", color=CB[2], ms=3, label="HFGAN-G branch")
        ax.plot(x, aq, "^-", color=CB[3], ms=3, label="QGE")
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("quartile of $w_{\\mathrm{HFGAN}}$")
        ax.set_ylabel("OA (%)" if j == 0 else "")
        ax.legend(frameon=False, fontsize=6)
        ax.grid(alpha=0.3)
    save(fig, "fig6_gate")


# ---------------------------------------------------------------- Fig. 7
def fig_qualitative(ds="ucmerced", n=4):
    from PIL import Image
    cache = os.path.join(C.RUNS, "cache")
    doms = ["lr_p1", "bic_p1", "edsr_p1", "hfgan_g_p1", "hr"]
    if not all(os.path.exists(os.path.join(cache, ds + "_" + d + ".npy")) for d in doms):
        return
    zz = os.path.join(C.RESULTS, "qgeweights_" + ds + "_p1_42.npz")
    if not os.path.exists(zz):
        return
    sp = C.get_splits(ds)
    z = np.load(zz)
    w, y = z["w"], z["y"]
    fixed = np.where((z["pred_qge"] == y) & (z["pred_gan"] != y))[0]
    rank = fixed[np.argsort(-w[fixed, 0])][:max(1, n // 2)]
    helped = np.where((z["pred_gan"] == y) & (z["pred_bic"] != y))[0]
    rank2 = helped[np.argsort(-w[helped, 2])][:n - len(rank)]
    sel = np.concatenate([rank, rank2])[:n].astype(int)
    if len(sel) == 0:
        return
    arrs = {d: np.load(os.path.join(cache, ds + "_" + d + ".npy"), mmap_mode="r") for d in doms}
    hr_size = C.DATASETS[ds]["hr"]
    fig, axes = plt.subplots(len(sel), len(doms), figsize=(COL2, 1.55 * len(sel)))
    axes = np.atleast_2d(axes)
    titles = ["LR input", "Bicubic", "EDSR", "HFGAN-G", "HR reference"]
    for r, s in enumerate(sel):
        gi = sp["test"][s]
        for c, d in enumerate(doms):
            im = np.asarray(arrs[d][gi])
            if im.shape[0] != hr_size:
                im = np.asarray(Image.fromarray(im).resize((hr_size, hr_size), Image.NEAREST))
            axes[r, c].imshow(im)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=7)
        axes[r, 0].set_ylabel("%s\n$w$=(%.2f, %.2f, %.2f)" %
                              (sp["classes"][y[s]], w[s, 0], w[s, 1], w[s, 2]), fontsize=6)
    save(fig, "fig7_qualitative_" + ds)


# ---------------------------------------------------------------- Fig. 8
def fig_matrix(ds="ucmerced"):
    fac = T.get(ds, {}).get("factorial")
    if not fac:
        return
    tr = ["hr", "bic_p1", "srcnn_p1", "edsr_p1", "hfgan_p_p1", "hfgan_g_p1"]
    te = ["hr", "bic_p1", "srcnn_p1", "edsr_p1", "hfgan_p_p1", "hfgan_g_p1",
          "bic_p2", "hfgan_g_p2"]
    M = np.full((len(tr), len(te)), np.nan)
    for i, a in enumerate(tr):
        for j, b in enumerate(te):
            v = fac.get(a, {}).get(b)
            if v:
                M[i, j] = v[0] * 100
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    lab = []
    for t in te:
        base = t.split("_p")[0]
        lab.append(PRETTY.get(base, base) + (" (P2)" if t.endswith("p2") else ""))
    ax.set_xticks(range(len(te)))
    ax.set_xticklabels(lab, rotation=35, ha="right")
    ax.set_yticks(range(len(tr)))
    ax.set_yticklabels([PRETTY.get(t.split("_p")[0], t) for t in tr])
    ax.set_xlabel("test domain")
    ax.set_ylabel("training domain")
    mid = np.nanmean(M)
    for i in range(len(tr)):
        for j in range(len(te)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, "%.1f" % M[i, j], ha="center", va="center",
                        fontsize=6, color="w" if M[i, j] < mid else "k")
    fig.colorbar(im, ax=ax, label="OA (%)", fraction=0.03)
    save(fig, "fig8_matrix_" + ds)


# ---------------------------------------------------------------- Fig. 9
def fig_qge_bars():
    dss = [d for d in ["ucmerced", "eurosat"] if "qge_p1" in T.get(d, {})]
    if not dss:
        return
    keys = ["single_bic", "single_edsr", "single_hfgan_g", "majority", "avg",
            "global_w", "stacking", "confmax", "arch_ens", "qge"]
    lbl = ["Bicubic", "EDSR", "HFGAN-G", "Majority", "Average", "Weighted",
           "Stacking", "Conf-max", "Arch. ens.", "QGE (ours)"]
    probe = T[dss[0]]["qge_p1"]
    probe = probe[list(probe)[0]]
    keep = [i for i, k in enumerate(keys) if k in probe]
    keys = [keys[i] for i in keep]
    lbl = [lbl[i] for i in keep]
    fig, axes = plt.subplots(1, len(dss), figsize=(COL2, 2.3), squeeze=False)
    for j, ds in enumerate(dss):
        ax = axes[0][j]
        lo = 100.0
        for pi, p in enumerate(["p1", "p2"]):
            blk = T[ds].get("qge_" + p)
            if not blk:
                continue
            mu = [np.mean([blk[s][k]["oa"] for s in blk]) * 100 for k in keys]
            sd = [np.std([blk[s][k]["oa"] for s in blk]) * 100 for k in keys]
            lo = min(lo, min(mu))
            x = np.arange(len(keys)) + (pi - 0.5) * 0.4
            ax.bar(x, mu, 0.38, yerr=sd, capsize=1.5,
                   color=[CB[3] if k == "qge" else CB[0] for k in keys],
                   alpha=1.0 if pi == 0 else 0.5,
                   label="P1 (bicubic)" if pi == 0 else "P2 (blur+noise+JPEG)")
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(lbl, rotation=40, ha="right", fontsize=6.4)
        ax.set_ylabel("OA (%)" if j == 0 else "")
        ax.set_title("UC Merced" if ds == "ucmerced" else "EuroSAT")
        ax.set_ylim(max(0.0, lo - 4), None)
        ax.grid(axis="y", alpha=0.3)
        if j == 0:
            ax.legend(frameon=False, fontsize=6, loc="best")
    save(fig, "fig9_qge")


if __name__ == "__main__":
    os.makedirs(C.FIGS, exist_ok=True)
    for f in [fig_framework, fig_confound, fig_pd_plane, fig_perclass, fig_corr,
              fig_gate, fig_qualitative, fig_matrix, fig_qge_bars]:
        try:
            f()
        except Exception as e:
            print("  SKIP", f.__name__, type(e).__name__, e)
