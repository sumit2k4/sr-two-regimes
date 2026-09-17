"""Train one SR model on one dataset under degradation protocol P1 (bicubic x4).

  python train_sr.py --dataset ucmerced --model hfgan_g --epochs 120
Checkpoints -> runs/sr/<dataset>_<model>.pt ; log -> runs/sr/<dataset>_<model>.log
"""
import os, sys, time, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, degrade as DG
from sr_models import build_sr, Discriminator, n_params

torch.backends.cudnn.benchmark = True

OUT = os.path.join(C.RUNS, "sr"); os.makedirs(OUT, exist_ok=True)


class VGGPerc(nn.Module):
    """VGG-19 relu5_4 feature L1, as in the Paper-2 objective."""
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg19(pretrained=True)
        self.f = nn.Sequential(*list(vgg.features.children())[:35]).eval()
        for p in self.f.parameters():
            p.requires_grad = False
        self.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        return F.l1_loss(self.f((sr - self.m) / self.s), self.f((hr - self.m) / self.s))


def load_pairs(dataset, idxs, hr_size, scale=4, mode="bicubic"):
    sp = C.get_splits(dataset)
    HR, LR = [], []
    for i in idxs:
        hr = DG.make_hr(Image.open(sp["paths"][i]).convert("RGB"), hr_size)
        lr = DG.degrade(hr, mode, scale, seed=i)
        HR.append(np.asarray(hr, np.uint8)); LR.append(np.asarray(lr, np.uint8))
    return np.stack(HR), np.stack(LR)


def crops(HR, LR, bs, p_lr, scale, rng):
    n = len(HR); ids = rng.randint(0, n, bs)
    hs, ls = [], []
    H = LR.shape[1]
    for i in ids:
        if H == p_lr:
            x = y = 0
        else:
            x = rng.randint(0, H - p_lr + 1); y = rng.randint(0, H - p_lr + 1)
        l = LR[i, y:y + p_lr, x:x + p_lr]
        h = HR[i, y * scale:(y + p_lr) * scale, x * scale:(x + p_lr) * scale]
        k = rng.randint(4); f = rng.randint(2)
        l = np.rot90(l, k); h = np.rot90(h, k)
        if f: l = l[:, ::-1]; h = h[:, ::-1]
        ls.append(l.copy()); hs.append(h.copy())
    t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float().div_(255.)
    return t(ls), t(hs)


@torch.no_grad()
def val_psnr(model, HR, LR, dev, p_lr, scale=4, limit=60):
    """Validation on centre crops at the training token count (cheap and free of
    any train/test positional-grid mismatch)."""
    model.eval(); ps = []
    for i in range(min(limit, len(HR))):
        H = LR.shape[1]; y = (H - p_lr) // 2
        lr = LR[i, y:y + p_lr, y:y + p_lr]
        hr = HR[i, y * scale:(y + p_lr) * scale, y * scale:(y + p_lr) * scale]
        t = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1)[None].float().div_(255.).to(dev)
        with torch.cuda.amp.autocast():
            sr = model(t).float().clamp(0, 1)
        ps.append(C.psnr(sr[0].permute(1, 2, 0).cpu().numpy(), hr / 255.0))
    model.train(); return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--iters_per_epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_minutes", type=float, default=90)
    a = ap.parse_args()

    cfg = C.DATASETS[a.dataset]; hr_size, scale = cfg["hr"], C.SCALE
    p_lr = min(32, cfg["lr"])
    C.set_seed(C.SPLIT_SEED)
    sp = C.get_splits(a.dataset)
    t0 = time.time()
    HRtr, LRtr = load_pairs(a.dataset, sp["train"], hr_size)
    HRva, LRva = load_pairs(a.dataset, sp["val"][:60], hr_size)
    dev = C.DEVICE
    model = build_sr(a.model, scale, img_size=p_lr).to(dev)
    tag = f"{a.dataset}_{a.model}"
    log = open(os.path.join(OUT, tag + ".log"), "w")

    def P(*s):
        print(*s); print(*s, file=log, flush=True)

    P(f"[{tag}] params={n_params(model):,} train={len(HRtr)} p_lr={p_lr} "
      f"hr={hr_size} data_load={time.time()-t0:.0f}s")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    use_perc = a.model in ("hfgan_p", "hfgan_g")
    use_adv = a.model == "hfgan_g"
    perc = VGGPerc().to(dev) if use_perc else None
    if use_adv:
        D = Discriminator().to(dev)
        optD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.9, 0.999))
        scalerD = torch.cuda.amp.GradScaler()
        bce = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(C.SPLIT_SEED)
    best, hist = -1, []
    for ep in range(1, a.epochs + 1):
        agg = np.zeros(4); 
        for it in range(a.iters_per_epoch):
            lr_b, hr_b = crops(HRtr, LRtr, a.batch, p_lr, scale, rng)
            lr_b, hr_b = lr_b.to(dev), hr_b.to(dev)
            with torch.cuda.amp.autocast():
                sr = model(lr_b)
                l_pix = F.l1_loss(sr, hr_b)
                l_per = perc(sr, hr_b) if use_perc else torch.zeros((), device=dev)
                if use_adv:
                    l_adv = bce(D(sr), torch.ones(lr_b.size(0), 1, device=dev))
                else:
                    l_adv = torch.zeros((), device=dev)
                loss = 1.0 * l_pix + 0.01 * l_per + 0.005 * l_adv
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if use_adv:
                with torch.cuda.amp.autocast():
                    dr = D(hr_b); df = D(sr.detach())
                    ld = bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df))
                optD.zero_grad(set_to_none=True)
                scalerD.scale(ld).backward(); scalerD.step(optD); scalerD.update()
            else:
                ld = torch.zeros(())
            agg += [float(l_pix), float(l_per), float(l_adv), float(ld)]
        sch.step()
        agg /= a.iters_per_epoch
        if ep % 5 == 0 or ep == a.epochs:
            v = val_psnr(model, HRva, LRva, dev, p_lr, scale)
            hist.append((ep, v))
            P(f"ep {ep:3d} pix {agg[0]:.4f} perc {agg[1]:.3f} adv {agg[2]:.3f} "
              f"D {agg[3]:.3f} valPSNR {v:.3f} [{(time.time()-t0)/60:.1f}m]")
            if v > best:
                best = v
                torch.save({"model": model.state_dict(), "ep": ep, "val_psnr": v,
                            "cfg": vars(a), "p_lr": p_lr}, os.path.join(OUT, tag + ".pt"))
                if use_adv:
                    torch.save(D.state_dict(), os.path.join(OUT, tag + "_D.pt"))
        if (time.time() - t0) / 60 > a.max_minutes:
            P(f"time cap reached at epoch {ep}"); break
    json.dump({"best_val_psnr": best, "hist": hist, "params": n_params(model)},
              open(os.path.join(OUT, tag + ".json"), "w"))
    P(f"[{tag}] done best val PSNR {best:.3f} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
