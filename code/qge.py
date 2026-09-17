"""Quality-Gated Ensemble (QGE).

Three classification branches see three reconstructions of the same scene that
sit at different points of the perception-distortion trade-off:
    v1 = bicubic upsample   (geometry faithful, texture free)
    v2 = EDSR               (distortion oriented)
    v3 = HFGAN-G            (perception oriented, may hallucinate)
A gate network reads a ground-truth-free reliability descriptor and emits
per-image mixing weights, so the classifier can fall back on the faithful view
whenever the perceptual view is untrustworthy.

The gate is fitted on the validation split only; the test split is untouched.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sr_models import Discriminator

CACHE = os.path.join(C.RUNS, "cache")
CLS = os.path.join(C.RUNS, "cls")
VIEWS = ["bic", "edsr", "hfgan_g"]


# ------------------------------------------------------------ descriptors
def _lap_var(x):
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=x.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=3).var(dim=(1, 2, 3))


def _hf_ratio(x):
    g = x.mean(1)
    f = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs()
    H, W = g.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device))
    r = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2).sqrt()
    hi = (r > 0.25 * min(H, W)).float()
    return (f * hi).sum((-2, -1)) / f.sum((-2, -1)).clamp_min(1e-8)


@torch.no_grad()
def image_features(ds, idx, protocol, disc=None, bs=32):
    """No-reference image statistics for each view plus cross-view distances."""
    dev = C.DEVICE
    arrs = {v: np.load(os.path.join(CACHE, ds + "_" + v + "_" + protocol + ".npy"),
                       mmap_mode="r") for v in VIEWS}
    feats = []
    for k in range(0, len(idx), bs):
        ids = np.asarray(idx[k:k + bs])
        X = {}
        for v in VIEWS:
            t = torch.from_numpy(np.ascontiguousarray(arrs[v][ids])).to(dev)
            X[v] = t.permute(0, 3, 1, 2).float().div_(255.)
        f = []
        for v in VIEWS:
            f.append(torch.log1p(_lap_var(X[v]) * 1e4))
            f.append(_hf_ratio(X[v]))
        f.append((X["edsr"] - X["bic"]).abs().mean((1, 2, 3)))
        f.append((X["hfgan_g"] - X["bic"]).abs().mean((1, 2, 3)))
        f.append((X["hfgan_g"] - X["edsr"]).abs().mean((1, 2, 3)))
        if disc is not None:
            for v in ["edsr", "hfgan_g"]:
                f.append(torch.sigmoid(disc(X[v])).squeeze(1))
        else:
            f.append(torch.zeros(len(ids), device=dev))
            f.append(torch.zeros(len(ids), device=dev))
        feats.append(torch.stack(f, 1).float().cpu())
    return torch.cat(feats).numpy()


def posterior_features(P):
    """Entropy and confidence per branch, plus pairwise Jensen-Shannon values."""
    eps = 1e-8
    out = []
    for k in range(P.shape[1]):
        p = P[:, k]
        out.append(-(p * np.log(p + eps)).sum(1))
        out.append(p.max(1))

    def kl(a, b):
        return (a * (np.log(a + eps) - np.log(b + eps))).sum(1)

    for i in range(P.shape[1]):
        for j in range(i + 1, P.shape[1]):
            m = 0.5 * (P[:, i] + P[:, j])
            out.append(0.5 * kl(P[:, i], m) + 0.5 * kl(P[:, j], m))
    return np.stack(out, 1)


# ------------------------------------------------------------------- gate
class Gate(nn.Module):
    def __init__(self, d_in, k, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(True),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(True),
                                 nn.Linear(hidden // 2, k))

    def forward(self, z):
        return torch.softmax(self.net(z), dim=1)


def _ece(p, y, bins=15):
    conf = p.max(1)
    pred = p.argmax(1)
    acc = (pred == y).astype(float)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def metrics(p, y, ncls):
    from sklearn.metrics import f1_score, cohen_kappa_score
    pred = p.argmax(1)
    return dict(oa=float((pred == y).mean()),
                mf1=float(f1_score(y, pred, average="macro",
                                   labels=list(range(ncls)), zero_division=0)),
                kappa=float(cohen_kappa_score(y, pred)),
                ece=_ece(p, y))


def mcnemar(a_correct, b_correct):
    from scipy.stats import binomtest
    n01 = int((a_correct & ~b_correct).sum())
    n10 = int(((~a_correct) & b_correct).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(binomtest(n01, n01 + n10, 0.5).pvalue), n01, n10


def softmax_np(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def load_branches(ds, seed, protocol, backbone=None):
    backbone = backbone or C.PRIMARY_BACKBONE
    """Posteriors of the matched-domain branch classifiers, shape (N, K, C)."""
    Pv, Pt = [], []
    z = None
    for v in VIEWS:
        f = os.path.join(CLS, ds + "_" + v + "_p1_" + backbone + "_" + str(seed) + ".npz")
        z = np.load(f)
        Pv.append(softmax_np(z["val_" + v + "_" + protocol]))
        Pt.append(softmax_np(z["test_" + v + "_" + protocol]))
    return np.stack(Pv, 1), np.stack(Pt, 1), z["y_val"], z["y_test"]


def load_arch_ensemble(ds, seed, protocol, view="hfgan_g",
                       backbones=("convnext_tiny", "efficientnet_b0", "densenet121")):
    """Posteriors of several architectures on the SAME super-resolved view: the
    conventional architecture ensemble. Returns None if the runs are absent."""
    Pv, Pt = [], []
    for b in backbones:
        f = os.path.join(CLS, ds + "_" + view + "_p1_" + b + "_" + str(seed) + ".npz")
        if not os.path.exists(f):
            return None, None
        z = np.load(f)
        Pv.append(softmax_np(z["val_" + view + "_" + protocol]))
        Pt.append(softmax_np(z["test_" + view + "_" + protocol]))
    return np.stack(Pv, 1), np.stack(Pt, 1)


def _fit_global_weights(Pv, yv, grid=21):
    best, bw = -1, np.ones(Pv.shape[1]) / Pv.shape[1]
    for a in np.linspace(0, 1, grid):
        for b in np.linspace(0, 1 - a, grid):
            w = np.array([a, b, max(0.0, 1 - a - b)])
            w = w / w.sum()
            acc = ((Pv * w[None, :, None]).sum(1).argmax(1) == yv).mean()
            if acc > best:
                best, bw = acc, w
    return bw


def _stacking(Pv, yv, Pt, yt, ncls):
    from sklearn.linear_model import LogisticRegression
    Xv = Pv.reshape(len(yv), -1)
    Xt = Pt.reshape(len(yt), -1)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xv, yv)
    proba = np.zeros((len(yt), ncls))
    proba[:, lr.classes_] = lr.predict_proba(Xt)
    return metrics(proba, yt, ncls), proba


def _train_gate(Zv, Pv, yv, Zt, Pt, alpha, beta, gamma, epochs, seed):
    dev = C.DEVICE
    C.set_seed(seed)
    zv = torch.tensor(Zv, dtype=torch.float32, device=dev)
    pv = torch.tensor(Pv, dtype=torch.float32, device=dev)
    yv_t = torch.tensor(np.asarray(yv), dtype=torch.long, device=dev)
    g = Gate(zv.shape[1], pv.shape[1]).to(dev)
    opt = torch.optim.Adam(g.parameters(), lr=3e-3, weight_decay=1e-4)
    n = len(yv)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    ce_k = -torch.log(pv[torch.arange(n), :, yv_t].clamp_min(1e-8))
    kl_gb = (pv[:, 2] * (torch.log(pv[:, 2] + 1e-8)
                         - torch.log(pv[:, 0] + 1e-8))).sum(1)
    for ep in range(epochs):
        rng.shuffle(idx)
        for k in range(0, n, 128):
            s = torch.tensor(idx[k:k + 128], dtype=torch.long, device=dev)
            w = g(zv[s])
            mix = (pv[s] * w[:, :, None]).sum(1).clamp_min(1e-8)
            loss = F.nll_loss(torch.log(mix), yv_t[s])
            loss = loss + alpha * (w * ce_k[s]).sum(1).mean()
            loss = loss - beta * (-(w * torch.log(w + 1e-8)).sum(1)).mean()
            loss = loss + gamma * (w[:, 2] * kl_gb[s]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    g.eval()
    with torch.no_grad():
        wt = g(torch.tensor(Zt, dtype=torch.float32, device=dev)).cpu().numpy()
    return (Pt * wt[:, :, None]).sum(1), wt


def run(ds, protocol="p1", seeds=None, alpha=0.3, beta=0.02, gamma=0.05,
        epochs=300, ablation=None):
    seeds = seeds or C.CLS_SEEDS
    sp = C.get_splits(ds)
    ncls = len(sp["classes"])
    dpath = os.path.join(C.RUNS, "sr", ds + "_hfgan_g_D.pt")
    disc = None
    if os.path.exists(dpath):
        disc = Discriminator().to(C.DEVICE)
        disc.load_state_dict(torch.load(dpath, map_location=C.DEVICE))
        disc.eval()
    cache_f = os.path.join(C.RESULTS, "imgfeat_" + ds + "_" + protocol + ".npz")
    if os.path.exists(cache_f):
        z = np.load(cache_f)
        Fv, Ft = z["val"], z["test"]
    else:
        Fv = image_features(ds, sp["val"], protocol, disc)
        Ft = image_features(ds, sp["test"], protocol, disc)
        np.savez_compressed(cache_f, val=Fv, test=Ft)
    out = {}
    for seed in seeds:
        Pv, Pt, yv, yt = load_branches(ds, seed, protocol)
        Zv = np.concatenate([Fv, posterior_features(Pv)], 1)
        Zt = np.concatenate([Ft, posterior_features(Pt)], 1)
        mu, sd = Zv.mean(0, keepdims=True), Zv.std(0, keepdims=True) + 1e-6
        Zv, Zt = (Zv - mu) / sd, (Zt - mu) / sd
        if ablation == "no_realism":
            Zv[:, 9:11] = 0.0
            Zt[:, 9:11] = 0.0
        if ablation == "no_crossview":
            Zv[:, 6:9] = 0.0
            Zt[:, 6:9] = 0.0
        if ablation == "no_posterior":
            Zv[:, 11:] = 0.0
            Zt[:, 11:] = 0.0
        a_, b_, g_ = alpha, beta, gamma
        if ablation == "no_branch_sup":
            a_ = 0.0
        if ablation == "no_halluc_pen":
            g_ = 0.0
        res = {}
        preds = {}
        for i, v in enumerate(VIEWS):
            res["single_" + v] = metrics(Pt[:, i], yt, ncls)
            preds["single_" + v] = Pt[:, i]
        preds["avg"] = Pt.mean(1)
        res["avg"] = metrics(preds["avg"], yt, ncls)
        vote = np.zeros_like(Pt[:, 0])
        for i in range(len(VIEWS)):
            vote[np.arange(len(yt)), Pt[:, i].argmax(1)] += 1.0
        preds["majority"] = vote / len(VIEWS) + 1e-6 * Pt.mean(1)
        res["majority"] = metrics(preds["majority"], yt, ncls)
        preds["confmax"] = Pt[np.arange(len(yt)), Pt.max(2).argmax(1)]
        res["confmax"] = metrics(preds["confmax"], yt, ncls)
        gw = _fit_global_weights(Pv, yv)
        preds["global_w"] = (Pt * gw[None, :, None]).sum(1)
        res["global_w"] = metrics(preds["global_w"], yt, ncls)
        res["global_w_weights"] = gw.tolist()
        Av, At = load_arch_ensemble(ds, seed, protocol)
        if At is not None:
            preds["arch_ens"] = At.mean(1)
            res["arch_ens"] = metrics(preds["arch_ens"], yt, ncls)
        m_st, p_st = _stacking(Pv, yv, Pt, yt, ncls)
        res["stacking"] = m_st
        preds["stacking"] = p_st
        pt, w_test = _train_gate(Zv, Pv, yv, Zt, Pt, a_, b_, g_, epochs, seed)
        res["qge"] = metrics(pt, yt, ncls)
        res["qge_mean_weights"] = w_test.mean(0).tolist()
        res["qge_weight_std"] = w_test.std(0).tolist()
        cand = [k for k in preds if k != "qge"]
        best_base = max(cand, key=lambda k: res[k]["oa"])
        p_val, n01, n10 = mcnemar(pt.argmax(1) == yt, preds[best_base].argmax(1) == yt)
        res["mcnemar_vs_best_baseline"] = dict(baseline=best_base, p=p_val,
                                               n01=n01, n10=n10)
        out[str(seed)] = res
        np.savez_compressed(os.path.join(C.RESULTS,
                            "qgeweights_" + ds + "_" + protocol + "_" + str(seed) + ".npz"),
                            w=w_test, y=yt, pred_qge=pt.argmax(1),
                            pred_bic=Pt[:, 0].argmax(1), pred_gan=Pt[:, 2].argmax(1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", default="p1")
    ap.add_argument("--ablation", default=None)
    a = ap.parse_args()
    r = run(a.dataset, a.protocol, ablation=a.ablation)
    suffix = "_" + a.ablation if a.ablation else ""
    name = "qge_" + a.dataset + "_" + a.protocol + suffix + ".json"
    json.dump(r, open(os.path.join(C.RESULTS, name), "w"), indent=1)
    summ = {}
    for s, v in r.items():
        summ[s] = {k: round(vv["oa"], 4) for k, vv in v.items()
                   if isinstance(vv, dict) and "oa" in vv}
    print(name, json.dumps(summ, indent=1))
