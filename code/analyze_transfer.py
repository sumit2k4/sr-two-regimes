"""The transfer regime: an HR-trained classifier applied to reconstructions.

This is the deployment scenario most super-resolution papers implicitly assume
(an existing archive-trained model is reused on enhanced imagery), and it
behaves completely differently from the matched-domain regime measured in
analyze.py. Everything here is computed from the same cached logits.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from analyze import load_run, oa, mean_std, SR_MODELS
from qge import mcnemar

BACKBONE = C.PRIMARY_BACKBONE


def transfer_block(ds, protocol="p1"):
    """Accuracy of the HR-trained classifier on every reconstruction domain."""
    doms = ["bic"] + SR_MODELS
    out = {}
    base_correct = None
    for m in doms:
        d = m + "_" + protocol
        vals, pvals = [], []
        for s in C.CLS_SEEDS:
            z = load_run(ds, "hr", BACKBONE, s)
            if z is None or ("test_" + d) not in z.files:
                continue
            vals.append(oa(z, d))
            if m != "bic":
                a = z["test_" + d].argmax(1) == z["y_test"]
                b = z["test_bic_" + protocol].argmax(1) == z["y_test"]
                pvals.append(mcnemar(a, b)[0])
        ms = mean_std(vals)
        if ms is None:
            continue
        out[m] = dict(acc=ms, p=float(np.median(pvals)) if pvals else None)
    if "bic" in out:
        for m in doms:
            if m in out:
                out[m]["gain"] = out[m]["acc"][0] - out["bic"]["acc"][0]
    # the matched-domain ceiling for reference
    hr = mean_std([oa(load_run(ds, "hr", BACKBONE, s), "hr") for s in C.CLS_SEEDS])
    out["_hr_matched"] = hr
    return out


def transfer_corr(ds, protocol="p1"):
    """Per-class correlation between the change in each reconstruction metric and
    the change in TRANSFER accuracy relative to the bicubic control."""
    from scipy.stats import spearmanr, pearsonr
    q = np.load(os.path.join(C.RESULTS, "srquality_" + ds + ".npz"))
    sp = C.get_splits(ds)
    y = np.asarray(sp["labels"])[sp["test"]]
    ncls = len(sp["classes"])
    zs = [load_run(ds, "hr", BACKBONE, s) for s in C.CLS_SEEDS]
    zs = [z for z in zs if z is not None]
    base = np.mean([[float((z["test_bic_" + protocol].argmax(1)[y == c] == c).mean())
                     for c in range(ncls)] for z in zs], 0)
    pts = []
    for m in SR_MODELS:
        k = "test_" + m + "_" + protocol
        if k not in zs[0].files:
            continue
        a = np.mean([[float((z[k].argmax(1)[y == c] == c).mean())
                      for c in range(ncls)] for z in zs], 0)
        for c in range(ncls):
            sel = y == c
            pts.append(dict(model=m, cls=int(c), d_acc=float(a[c] - base[c]),
                            d_psnr=float(q[m + "_" + protocol + "_test_psnr"][sel].mean()
                                         - q["bic_" + protocol + "_test_psnr"][sel].mean()),
                            d_ssim=float(q[m + "_" + protocol + "_test_ssim"][sel].mean()
                                         - q["bic_" + protocol + "_test_ssim"][sel].mean()),
                            d_lpips=float(q[m + "_" + protocol + "_test_lpips"][sel].mean()
                                          - q["bic_" + protocol + "_test_lpips"][sel].mean())))
    res = {"points": pts}
    for k in ["d_psnr", "d_ssim", "d_lpips"]:
        x = np.array([p[k] for p in pts])
        yv = np.array([p["d_acc"] for p in pts])
        rs, ps = spearmanr(x, yv)
        rp, pp = pearsonr(x, yv)
        res[k] = dict(spearman=float(rs), spearman_p=float(ps),
                      pearson=float(rp), pearson_p=float(pp), n=len(x))
    return res


def main():
    out = {}
    for ds in ["ucmerced", "eurosat"]:
        try:
            out[ds] = dict(transfer=transfer_block(ds),
                           transfer_corr=transfer_corr(ds))
        except Exception as e:
            print("skip", ds, type(e).__name__, e)
    json.dump(out, open(os.path.join(C.RESULTS, "transfer.json"), "w"), indent=1)
    for ds, d in out.items():
        print("=" * 66)
        print(ds.upper(), " HR-trained classifier, matched HR accuracy %.2f"
              % (d["transfer"]["_hr_matched"][0] * 100))
        for m in ["bic"] + SR_MODELS:
            if m not in d["transfer"]:
                continue
            r = d["transfer"][m]
            print("  %-9s %.2f +- %.2f   gain vs bicubic %+.2f   p=%s"
                  % (m, r["acc"][0] * 100, r["acc"][1] * 100, r["gain"] * 100,
                     ("%.3g" % r["p"]) if r["p"] is not None else "--"))
        for k in ["d_psnr", "d_ssim", "d_lpips"]:
            v = d["transfer_corr"][k]
            print("   corr %-8s rho=%+.3f (p=%.2g)  n=%d"
                  % (k, v["spearman"], v["spearman_p"], v["n"]))


if __name__ == "__main__":
    main()
