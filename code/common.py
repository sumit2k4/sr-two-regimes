"""Shared protocol constants, deterministic splits, metrics and I/O helpers.

Protocol (frozen, quoted verbatim in the manuscript):
  * scale x4, stratified 70/15/15 split per dataset, split seed = 42
  * SR models are trained ONLY on the training split -> the classification test
    split is never seen by any super-resolution model.
  * classifier seeds = 42, 123, 999
"""
import os, json, random, hashlib
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figs")
for d in (RUNS, RESULTS, FIGS):
    os.makedirs(d, exist_ok=True)

SPLIT_SEED = 42
CLS_SEEDS = [42, 123, 999]
SCALE = 4

# backbone whose results are reported in the manuscript; the
# ResNet-18 runs are retained as a backbone-robustness check
PRIMARY_BACKBONE = os.environ.get("P3_BACKBONE", "convnext_tiny")

DATASETS = {
    "ucmerced": dict(hr=256, lr=64, root=os.path.join(DATA, "ucmerced")),
    "eurosat":  dict(hr=64,  lr=16, root=os.path.join(DATA, "eurosat")),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def list_dataset(name):
    """Return (paths, labels, class_names) sorted deterministically."""
    root = DATASETS[name]["root"]
    classes = sorted([d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d))])
    paths, labels = [], []
    for ci, c in enumerate(classes):
        fs = sorted(os.listdir(os.path.join(root, c)))
        for f in fs:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                paths.append(os.path.join(root, c, f)); labels.append(ci)
    return paths, labels, classes


def _absolutise(sp):
    """Split files store paths relative to the project root so that they are
    portable; resolve them against this checkout on load."""
    sp = dict(sp)
    sp["paths"] = [p if os.path.isabs(p) else os.path.join(ROOT, p)
                   for p in sp["paths"]]
    return sp


def get_splits(name):
    """Stratified 70/15/15, cached to results/splits_<name>.json."""
    cache = os.path.join(RESULTS, f"splits_{name}.json")
    if os.path.exists(cache):
        return _absolutise(json.load(open(cache)))
    paths, labels, classes = list_dataset(name)
    rng = np.random.RandomState(SPLIT_SEED)
    idx_by_c = {}
    for i, l in enumerate(labels):
        idx_by_c.setdefault(l, []).append(i)
    tr, va, te = [], [], []
    for c, idxs in sorted(idx_by_c.items()):
        idxs = np.array(idxs); rng.shuffle(idxs)
        n = len(idxs); n_tr = int(round(0.70 * n)); n_va = int(round(0.15 * n))
        tr += idxs[:n_tr].tolist(); va += idxs[n_tr:n_tr + n_va].tolist()
        te += idxs[n_tr + n_va:].tolist()
    out = dict(classes=classes,
               paths=[os.path.relpath(p, ROOT).replace("\\", "/") for p in paths],
               labels=labels,
               train=sorted(tr), val=sorted(va), test=sorted(te))
    json.dump(out, open(cache, "w"))
    return _absolutise(out)


# ------------------------------------------------------------------ metrics
def psnr(a, b, data_range=1.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def _ssim_ch(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    """Gaussian-window SSIM on a single channel, float in [0,1]."""
    from scipy.ndimage import gaussian_filter
    mu_x = gaussian_filter(x, 1.5); mu_y = gaussian_filter(y, 1.5)
    xx = gaussian_filter(x * x, 1.5) - mu_x ** 2
    yy = gaussian_filter(y * y, 1.5) - mu_y ** 2
    xy = gaussian_filter(x * y, 1.5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (xx + yy + C2)
    return float(np.mean(num / den))


def ssim(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.ndim == 2:
        return _ssim_ch(a, b)
    return float(np.mean([_ssim_ch(a[..., c], b[..., c]) for c in range(a.shape[-1])]))


def load_img(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def md5(path, n=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b: break
            h.update(b)
    return h.hexdigest()
