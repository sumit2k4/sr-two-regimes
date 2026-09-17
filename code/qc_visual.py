"""Quick visual QC of the cached super-resolution domains: checks that tiled
inference leaves no seams and that each model looks the way its objective
predicts. Writes figs/qc_<dataset>.png (not a manuscript figure)."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CACHE = os.path.join(C.RUNS, "cache")


def main(ds="ucmerced", n=3):
    doms = [d for d in ["lr_p1", "bic_p1", "srcnn_p1", "edsr_p1",
                        "hfgan_p_p1", "hfgan_g_p1", "hr"]
            if os.path.exists(os.path.join(CACHE, ds + "_" + d + ".npy"))]
    sp = C.get_splits(ds)
    arrs = {d: np.load(os.path.join(CACHE, ds + "_" + d + ".npy"), mmap_mode="r")
            for d in doms}
    rng = np.random.RandomState(0)
    sel = rng.choice(sp["test"], n, replace=False)
    hr_size = C.DATASETS[ds]["hr"]
    fig, axes = plt.subplots(n, len(doms), figsize=(2.1 * len(doms), 2.2 * n))
    axes = np.atleast_2d(axes)
    for r, gi in enumerate(sel):
        for c, d in enumerate(doms):
            im = np.asarray(arrs[d][gi])
            if im.shape[0] != hr_size:
                from PIL import Image
                im = np.asarray(Image.fromarray(im).resize((hr_size, hr_size), Image.NEAREST))
            axes[r, c].imshow(im)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(d, fontsize=8)
            if d not in ("hr", "lr_p1"):
                hr = np.asarray(arrs["hr"][gi]).astype(np.float64) / 255.0
                axes[r, c].set_xlabel("PSNR %.2f" % C.psnr(im.astype(np.float64) / 255.0, hr),
                                      fontsize=7)
        axes[r, 0].set_ylabel(sp["classes"][sp["labels"][gi]], fontsize=7)
    fig.tight_layout(pad=0.3)
    p = os.path.join(C.FIGS, "qc_" + ds + ".png")
    fig.savefig(p, dpi=110)
    print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ucmerced")
