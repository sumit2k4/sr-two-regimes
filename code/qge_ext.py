"""Extended quality-gated ensemble.

The three-view gate and the conventional architecture ensemble draw on two
different sources of diversity: reconstructions of different fidelity, and
backbones with different inductive biases. Neither subsumes the other, and the
gate does not care where its experts come from. This module builds the union,

    (bicubic, ResNet-18), (EDSR, ResNet-18), (HFGAN-G, ResNet-18),
    (HFGAN-G, EfficientNet-B0), (HFGAN-G, DenseNet-121),

and gates over all five. It reuses classifier runs that already exist, so it
costs no additional training.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import qge
from qge import (CLS, softmax_np, posterior_features, image_features, metrics,
                 mcnemar, _train_gate, _stacking, Discriminator)

BRANCHES = [("bic", "resnet18"), ("edsr", "resnet18"), ("hfgan_g", "resnet18"),
            ("hfgan_g", "efficientnet_b0"), ("hfgan_g", "densenet121")]
LABELS = ["bicubic/R18", "EDSR/R18", "HFGAN-G/R18", "HFGAN-G/EffB0",
          "HFGAN-G/DN121"]


def load_branches_ext(ds, seed, protocol):
    Pv, Pt, z = [], [], None
    for view, bb in BRANCHES:
        f = os.path.join(CLS, ds + "_" + view + "_p1_" + bb + "_" + str(seed) + ".npz")
        if not os.path.exists(f):
            return None, None, None, None
        z = np.load(f)
        Pv.append(softmax_np(z["val_" + view + "_" + protocol]))
        Pt.append(softmax_np(z["test_" + view + "_" + protocol]))
    return np.stack(Pv, 1), np.stack(Pt, 1), z["y_val"], z["y_test"]


def fit_global_weights_simplex(Pv, yv, n=4000, seed=0):
    """Dirichlet random search over the simplex; the grid search used for three
    branches does not scale to five."""
    rng = np.random.RandomState(seed)
    K = Pv.shape[1]
    best, bw = -1.0, np.ones(K) / K
    cands = np.vstack([np.ones((1, K)) / K, np.eye(K),
                       rng.dirichlet(np.ones(K), size=n)])
    for w in cands:
        acc = ((Pv * w[None, :, None]).sum(1).argmax(1) == yv).mean()
        if acc > best:
            best, bw = acc, w
    return bw


def run(ds, protocol="p1", seeds=None, alpha=0.3, beta=0.02, gamma=0.05,
        epochs=300):
    seeds = seeds or C.CLS_SEEDS
    sp = C.get_splits(ds)
    ncls = len(sp["classes"])
    cache_f = os.path.join(C.RESULTS, "imgfeat_" + ds + "_" + protocol + ".npz")
    if os.path.exists(cache_f):
        z = np.load(cache_f)
        Fv, Ft = z["val"], z["test"]
    else:
        dpath = os.path.join(C.RUNS, "sr", ds + "_hfgan_g_D.pt")
        disc = None
        if os.path.exists(dpath):
            disc = Discriminator().to(C.DEVICE)
            disc.load_state_dict(torch.load(dpath, map_location=C.DEVICE))
            disc.eval()
        Fv = image_features(ds, sp["val"], protocol, disc)
        Ft = image_features(ds, sp["test"], protocol, disc)
        np.savez_compressed(cache_f, val=Fv, test=Ft)
    out = {}
    for seed in seeds:
        Pv, Pt, yv, yt = load_branches_ext(ds, seed, protocol)
        if Pv is None:
            print("missing runs for seed", seed)
            continue
        Zv = np.concatenate([Fv, posterior_features(Pv)], 1)
        Zt = np.concatenate([Ft, posterior_features(Pt)], 1)
        mu, sd = Zv.mean(0, keepdims=True), Zv.std(0, keepdims=True) + 1e-6
        Zv, Zt = (Zv - mu) / sd, (Zt - mu) / sd
        res, preds = {}, {}
        for i, lab in enumerate(LABELS):
            res["single_" + str(i)] = metrics(Pt[:, i], yt, ncls)
        preds["avg"] = Pt.mean(1)
        res["avg"] = metrics(preds["avg"], yt, ncls)
        vote = np.zeros_like(Pt[:, 0])
        for i in range(Pt.shape[1]):
            vote[np.arange(len(yt)), Pt[:, i].argmax(1)] += 1.0
        preds["majority"] = vote / Pt.shape[1] + 1e-6 * Pt.mean(1)
        res["majority"] = metrics(preds["majority"], yt, ncls)
        preds["confmax"] = Pt[np.arange(len(yt)), Pt.max(2).argmax(1)]
        res["confmax"] = metrics(preds["confmax"], yt, ncls)
        gw = fit_global_weights_simplex(Pv, yv)
        preds["global_w"] = (Pt * gw[None, :, None]).sum(1)
        res["global_w"] = metrics(preds["global_w"], yt, ncls)
        res["global_w_weights"] = gw.tolist()
        m_st, p_st = _stacking(Pv, yv, Pt, yt, ncls)
        res["stacking"] = m_st
        preds["stacking"] = p_st
        pt, wt = _train_gate(Zv, Pv, yv, Zt, Pt, alpha, beta, gamma, epochs, seed)
        res["qge_ext"] = metrics(pt, yt, ncls)
        res["qge_ext_mean_weights"] = wt.mean(0).tolist()
        res["qge_ext_weight_std"] = wt.std(0).tolist()
        best_base = max(preds, key=lambda k: res[k]["oa"])
        p_val, n01, n10 = mcnemar(pt.argmax(1) == yt, preds[best_base].argmax(1) == yt)
        res["mcnemar_vs_best_baseline"] = dict(baseline=best_base, p=p_val,
                                               n01=n01, n10=n10)
        out[str(seed)] = res
        np.savez_compressed(os.path.join(C.RESULTS, "qgeextw_" + ds + "_" + protocol
                                         + "_" + str(seed) + ".npz"),
                            w=wt, y=yt, pred=pt.argmax(1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", default="p1")
    a = ap.parse_args()
    r = run(a.dataset, a.protocol)
    name = "qgeext_" + a.dataset + "_" + a.protocol + ".json"
    json.dump(r, open(os.path.join(C.RESULTS, name), "w"), indent=1)
    summ = {}
    for s, v in r.items():
        summ[s] = {k: round(vv["oa"], 4) for k, vv in v.items()
                   if isinstance(vv, dict) and "oa" in vv}
    print(name, json.dumps(summ, indent=1))
