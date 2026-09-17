"""Materialise every input domain used by the study as a uint8 .npy cache.

Domains (per dataset):
  hr                     ground-truth high resolution (oracle)
  lr_p1 / lr_p2          low resolution, at LR size (the *uncontrolled* input)
  bic_p1 / bic_p2        LR bicubically upsampled to HR size (the CONTROL)
  <srmodel>_p1/_p2       SR reconstruction at HR size
P1 = bicubic x4, P2 = blur+noise+JPEG x4 (degradation shift, unseen in training).
"""
import os, sys, json, argparse, time
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, degrade as DG
from sr_models import build_sr, tiled_sr

CACHE = os.path.join(C.RUNS, "cache"); os.makedirs(CACHE, exist_ok=True)
SR_MODELS = ["srcnn", "edsr", "hfgan_p", "hfgan_g"]
PROTOCOLS = ["p1", "p2"]
PMODE = {"p1": "bicubic", "p2": "blur_noise"}


def path(ds, dom):
    return os.path.join(CACHE, f"{ds}_{dom}.npy")


def build_base(ds):
    cfg = C.DATASETS[ds]; hr_size = cfg["hr"]
    sp = C.get_splits(ds); n = len(sp["paths"])
    need = [d for d in ["hr", "lr_p1", "lr_p2", "bic_p1", "bic_p2"] if not os.path.exists(path(ds, d))]
    if not need:
        return
    HR = np.zeros((n, hr_size, hr_size, 3), np.uint8)
    LR = {p: np.zeros((n, cfg["lr"], cfg["lr"], 3), np.uint8) for p in PROTOCOLS}
    BIC = {p: np.zeros_like(HR) for p in PROTOCOLS}
    for i, p in enumerate(sp["paths"]):
        hr = DG.make_hr(Image.open(p).convert("RGB"), hr_size)
        HR[i] = np.asarray(hr, np.uint8)
        for pr in PROTOCOLS:
            lr = DG.degrade(hr, PMODE[pr], C.SCALE, seed=i)
            LR[pr][i] = np.asarray(lr, np.uint8)
            BIC[pr][i] = np.asarray(DG.bicubic_up(lr, C.SCALE), np.uint8)
        if i % 2000 == 0:
            print("  base", i, "/", n, flush=True)
    np.save(path(ds, "hr"), HR)
    for pr in PROTOCOLS:
        np.save(path(ds, f"lr_{pr}"), LR[pr]); np.save(path(ds, f"bic_{pr}"), BIC[pr])
    print("  base domains written")


@torch.no_grad()
def run_sr(ds, model_name, pr, batch=8):
    dom = f"{model_name}_{pr}"
    if os.path.exists(path(ds, dom)):
        print("  cached", dom); return
    cfg = C.DATASETS[ds]
    ck = torch.load(os.path.join(C.RUNS, "sr", f"{ds}_{model_name}.pt"), map_location="cpu")
    m = build_sr(model_name, C.SCALE, img_size=ck["p_lr"]).to(C.DEVICE)
    m.load_state_dict(ck["model"]); m.eval()
    LR = np.load(path(ds, f"lr_{pr}"), mmap_mode="r")
    n = len(LR); out = np.zeros((n, cfg["hr"], cfg["hr"], 3), np.uint8)
    tile = ck["p_lr"]
    need_tile = model_name.startswith("hfgan") and cfg["lr"] > tile
    bs = 4 if need_tile else batch
    t0 = time.time()
    for i in range(0, n, bs):
        x = torch.from_numpy(np.ascontiguousarray(LR[i:i + bs])).permute(0, 3, 1, 2).float().div_(255.).to(C.DEVICE)
        with torch.cuda.amp.autocast():
            y = (tiled_sr(m, x, tile=tile) if need_tile else m(x)).float().clamp(0, 1)
        out[i:i + bs] = (y.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        if i % (200 * bs) == 0:
            print(f"  {dom} {i}/{n} [{time.time()-t0:.0f}s]", flush=True)
    np.save(path(ds, dom), out)
    print(f"  {dom} done in {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default=",".join(SR_MODELS))
    a = ap.parse_args()
    print("[cache]", a.dataset)
    build_base(a.dataset)
    for mn in a.models.split(","):
        if not mn: continue
        for pr in PROTOCOLS:
            run_sr(a.dataset, mn, pr)


if __name__ == "__main__":
    main()
