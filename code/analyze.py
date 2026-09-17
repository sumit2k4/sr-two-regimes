"""Assemble every table of the manuscript from the cached run artefacts."""
import os
import sys
import json
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CLS = os.path.join(C.RUNS, "cls")
SRD = os.path.join(C.RUNS, "sr")
SR_MODELS = ["srcnn", "edsr", "hfgan_p", "hfgan_g"]
PRETTY = {"bic": "Bicubic", "srcnn": "SRCNN", "edsr": "EDSR",
          "hfgan_p": "HFGAN-P", "hfgan_g": "HFGAN-G", "hr": "HR (oracle)",
          "lr": "LR (native)"}


def softmax_np(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def load_run(ds, dom, backbone, seed):
    f = os.path.join(CLS, ds + "_" + dom + "_" + backbone + "_" + str(seed) + ".npz")
    return np.load(f) if os.path.exists(f) else None


def oa(z, test_dom):
    if z is None:
        return None
    k = "test_" + test_dom
    if k not in z.files:
        return None
    return float((z[k].argmax(1) == z["y_test"]).mean())


def mean_std(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return float(np.mean(v)), float(np.std(v))


def fmt(ms, mult=100, nd=2):
    if ms is None:
        return "--"
    return ("%." + str(nd) + "f $\\pm$ %." + str(nd) + "f") % (ms[0] * mult, ms[1] * mult)


# ------------------------------------------------------------------ tables
def table_factorial(ds, backbone=None):
    backbone = backbone or C.PRIMARY_BACKBONE
    train_doms = ["hr"] + ["bic_p1"] + [m + "_p1" for m in SR_MODELS] + ["lr_p1"]
    test_doms = ["hr", "bic_p1"] + [m + "_p1" for m in SR_MODELS] + \
                ["bic_p2"] + [m + "_p2" for m in SR_MODELS] + ["lr_p1", "lr_p2"]
    out = {}
    for td in train_doms:
        row = {}
        for ed in test_doms:
            vals = []
            for s in C.CLS_SEEDS:
                z = load_run(ds, td, backbone, s)
                vals.append(oa(z, ed) if z is not None else None)
            row[ed] = mean_std(vals)
        out[td] = row
    return out


def table_confound(ds, backbone=None):
    backbone = backbone or C.PRIMARY_BACKBONE
    """The core methodological result: naive vs controlled super-resolution gain."""
    rows = {}
    lr_acc = mean_std([oa(load_run(ds, "lr_p1", backbone, s), "lr_p1") for s in C.CLS_SEEDS])
    bic_acc = mean_std([oa(load_run(ds, "bic_p1", backbone, s), "bic_p1") for s in C.CLS_SEEDS])
    hr_acc = mean_std([oa(load_run(ds, "hr", backbone, s), "hr") for s in C.CLS_SEEDS])
    for m in SR_MODELS:
        sr = mean_std([oa(load_run(ds, m + "_p1", backbone, s), m + "_p1") for s in C.CLS_SEEDS])
        if sr is None:
            continue
        rows[m] = dict(sr=sr,
                       naive_gain=(sr[0] - lr_acc[0]) if lr_acc else None,
                       controlled_gain=sr[0] - bic_acc[0],
                       oracle_gap=hr_acc[0] - sr[0])
        pv = []
        for s in C.CLS_SEEDS:
            za = load_run(ds, m + "_p1", backbone, s)
            zb = load_run(ds, "bic_p1", backbone, s)
            if za is None or zb is None:
                continue
            a = za["test_" + m + "_p1"].argmax(1) == za["y_test"]
            b = zb["test_bic_p1"].argmax(1) == zb["y_test"]
            from qge import mcnemar
            pv.append(mcnemar(a, b)[0])
        rows[m]["mcnemar_p_vs_bic"] = float(np.median(pv)) if pv else None
    return dict(lr=lr_acc, bic=bic_acc, hr=hr_acc, models=rows)


def table_perclass(ds, model="hfgan_g", backbone=None):
    backbone = backbone or C.PRIMARY_BACKBONE
    sp = C.get_splits(ds)
    classes = sp["classes"]
    y = None
    accs = {}
    for dom in ["bic_p1", model + "_p1", "hr"]:
        per = []
        for s in C.CLS_SEEDS:
            z = load_run(ds, dom, backbone, s)
            if z is None:
                continue
            y = z["y_test"]
            pred = z["test_" + dom].argmax(1)
            per.append([float((pred[y == c] == c).mean()) for c in range(len(classes))])
        accs[dom] = np.mean(per, 0) if per else None
    if accs["bic_p1"] is None or accs[model + "_p1"] is None:
        return None
    if accs["hr"] is None:
        accs["hr"] = np.full_like(accs["bic_p1"], np.nan)
    delta = accs[model + "_p1"] - accs["bic_p1"]
    order = np.argsort(-delta)
    return dict(classes=classes, bic=accs["bic_p1"].tolist(),
                sr=accs[model + "_p1"].tolist(), hr=accs["hr"].tolist(),
                delta=delta.tolist(), order=order.tolist())


def table_quality_vs_gain(ds, backbone=None):
    backbone = backbone or C.PRIMARY_BACKBONE
    """Per-class reconstruction-quality change vs per-class accuracy change,
    pooled over all SR models -> the correlation study."""
    from scipy.stats import spearmanr, pearsonr
    q = np.load(os.path.join(C.RESULTS, "srquality_" + ds + ".npz"))
    sp = C.get_splits(ds)
    y = np.asarray(sp["labels"])[sp["test"]]
    ncls = len(sp["classes"])
    pts = []
    zb = [load_run(ds, "bic_p1", backbone, s) for s in C.CLS_SEEDS]
    zb = [z for z in zb if z is not None]
    base = np.mean([[float((z["test_bic_p1"].argmax(1)[y == c] == c).mean())
                     for c in range(ncls)] for z in zb], 0)
    for m in SR_MODELS:
        zs = [load_run(ds, m + "_p1", backbone, s) for s in C.CLS_SEEDS]
        zs = [z for z in zs if z is not None]
        if not zs:
            continue
        a = np.mean([[float((z["test_" + m + "_p1"].argmax(1)[y == c] == c).mean())
                      for c in range(ncls)] for z in zs], 0)
        for c in range(ncls):
            sel = y == c
            pts.append(dict(model=m, cls=int(c),
                            d_acc=float(a[c] - base[c]),
                            d_psnr=float(q[m + "_p1_test_psnr"][sel].mean()
                                         - q["bic_p1_test_psnr"][sel].mean()),
                            d_ssim=float(q[m + "_p1_test_ssim"][sel].mean()
                                         - q["bic_p1_test_ssim"][sel].mean()),
                            d_lpips=float(q[m + "_p1_test_lpips"][sel].mean()
                                          - q["bic_p1_test_lpips"][sel].mean())))
    res = {"points": pts}
    for k in ["d_psnr", "d_ssim", "d_lpips"]:
        x = np.array([p[k] for p in pts])
        yv = np.array([p["d_acc"] for p in pts])
        rs, ps = spearmanr(x, yv)
        rp, pp = pearsonr(x, yv)
        res[k] = dict(spearman=float(rs), spearman_p=float(ps),
                      pearson=float(rp), pearson_p=float(pp), n=len(x))
    return res


def table_sr_quality(ds):
    f = os.path.join(C.RESULTS, "srquality_" + ds + ".json")
    q = json.load(open(f)) if os.path.exists(f) else {}
    for m in SR_MODELS:
        j = os.path.join(SRD, ds + "_" + m + ".json")
        if os.path.exists(j) and (m + "_p1") in q:
            q[m + "_p1"]["params"] = json.load(open(j))["params"]
    return q


def main():
    all_out = {}
    for ds in ["ucmerced", "eurosat"]:
        if not os.path.exists(os.path.join(C.RUNS, "cache", ds + "_hr.npy")):
            continue
        d = {}
        for key, fn in [("factorial", lambda: table_factorial(ds)),
                        ("confound", lambda: table_confound(ds)),
                        ("perclass_hfgan_g", lambda: table_perclass(ds, "hfgan_g")),
                        ("perclass_edsr", lambda: table_perclass(ds, "edsr")),
                        ("sr_quality", lambda: table_sr_quality(ds)),
                        ("quality_vs_gain", lambda: table_quality_vs_gain(ds))]:
            try:
                d[key] = fn()
            except Exception as e:
                print("  SKIP", ds, key, type(e).__name__, e)
        for p in ["p1", "p2"]:
            f = os.path.join(C.RESULTS, "qge_" + ds + "_" + p + ".json")
            if os.path.exists(f):
                d["qge_" + p] = json.load(open(f))
        for ab in ["no_realism", "no_crossview", "no_posterior",
                   "no_branch_sup", "no_halluc_pen"]:
            f = os.path.join(C.RESULTS, "qge_" + ds + "_p1_" + ab + ".json")
            if os.path.exists(f):
                d["qge_ablation_" + ab] = json.load(open(f))
        all_out[ds] = d
    json.dump(all_out, open(os.path.join(C.RESULTS, "all_tables.json"), "w"), indent=1)
    for ds, d in all_out.items():
        if not d.get("confound"):
            continue
        print("=" * 70)
        print(ds.upper())
        cf = d["confound"]
        print("  LR-native %s | BIC %s | HR %s" % (fmt(cf["lr"]), fmt(cf["bic"]), fmt(cf["hr"])))
        for m, r in cf["models"].items():
            ng = ("%+.2f" % (r["naive_gain"] * 100)) if r.get("naive_gain") is not None else "--"
            print("   %-9s OA %s  naive %s  controlled %+.2f  oracle gap %.2f  p=%s"
                  % (PRETTY[m], fmt(r["sr"]), ng,
                     r["controlled_gain"] * 100, r["oracle_gap"] * 100,
                     ("%.3g" % r["mcnemar_p_vs_bic"]) if r["mcnemar_p_vs_bic"] else "--"))
        if "quality_vs_gain" in d:
            for k in ["d_psnr", "d_ssim", "d_lpips"]:
                v = d["quality_vs_gain"][k]
                print("   corr %-7s rho=%+.3f (p=%.3g) r=%+.3f n=%d"
                      % (k, v["spearman"], v["spearman_p"], v["pearson"], v["n"]))
        for p in ["p1", "p2"]:
            if "qge_" + p in d:
                r = d["qge_" + p]
                keys = [k for k in r[list(r)[0]] if isinstance(r[list(r)[0]][k], dict)
                        and "oa" in r[list(r)[0]][k]]
                print("  QGE block", p)
                for k in keys:
                    v = [r[s][k]["oa"] for s in r]
                    f1 = [r[s][k]["mf1"] for s in r]
                    print("    %-16s OA %.2f+-%.2f  mF1 %.2f+-%.2f"
                          % (k, np.mean(v) * 100, np.std(v) * 100,
                             np.mean(f1) * 100, np.std(f1) * 100))


if __name__ == "__main__":
    main()
