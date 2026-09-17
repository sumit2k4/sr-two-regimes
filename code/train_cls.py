"""Train one scene classifier on one input domain and dump its logits on every
domain, so the full train-domain x test-domain matrix can be assembled offline.

  python train_cls.py --dataset ucmerced --domain bic_p1 --backbone resnet18 --seed 42
Output: runs/cls/<dataset>_<domain>_<backbone>_<seed>.npz
"""
import os, sys, time, json, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from cls_models import build_cls, IMNET_MEAN, IMNET_STD, MIN_INPUT

torch.backends.cudnn.benchmark = True
OUT = os.path.join(C.RUNS, "cls"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(C.RUNS, "cache")


def dom_path(ds, dom):
    return os.path.join(CACHE, f"{ds}_{dom}.npy")


def all_domains(ds):
    pref = f"{ds}_"
    return sorted(f[len(pref):-4] for f in os.listdir(CACHE)
                  if f.startswith(pref) and f.endswith(".npy"))


class GPUBatcher:
    def __init__(self, arr, labels, idx, bs, train, dev):
        self.x = arr; self.y = np.asarray(labels)[idx]; self.idx = np.asarray(idx)
        self.bs = bs; self.train = train; self.dev = dev
        self.mean = IMNET_MEAN.to(dev); self.std = IMNET_STD.to(dev)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def epoch(self, rng=None):
        order = np.arange(len(self.idx))
        if self.train: rng.shuffle(order)
        for k in range(0, len(order), self.bs):
            sel = order[k:k + self.bs]
            ids = self.idx[sel]
            xb = torch.from_numpy(np.ascontiguousarray(self.x[ids])).to(self.dev)
            xb = xb.permute(0, 3, 1, 2).float().div_(255.)
            if self.train:
                if rng.rand() < 0.5: xb = torch.flip(xb, [3])
                k90 = rng.randint(4)
                if k90: xb = torch.rot90(xb, k90, [2, 3])
            xb = (xb - self.mean) / self.std
            yield xb, torch.from_numpy(self.y[sel]).long().to(self.dev)


@torch.no_grad()
def logits_on(model, arr, idx, dev, bs=64):
    model.eval(); out = []
    mean, std = IMNET_MEAN.to(dev), IMNET_STD.to(dev)
    for k in range(0, len(idx), bs):
        ids = np.asarray(idx[k:k + bs])
        xb = torch.from_numpy(np.ascontiguousarray(arr[ids])).to(dev)
        xb = ((xb.permute(0, 3, 1, 2).float().div_(255.)) - mean) / std
        with torch.cuda.amp.autocast():
            out.append(model(xb).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--eval_domains", default="all")
    a = ap.parse_args()

    tag = f"{a.dataset}_{a.domain}_{a.backbone}_{a.seed}"
    dst = os.path.join(OUT, tag + ".npz")
    if os.path.exists(dst):
        print("exists", tag); return
    sp = C.get_splits(a.dataset)
    labels = sp["labels"]; ncls = len(sp["classes"])
    bs = a.bs or (32 if a.dataset == "ucmerced" else 128)
    C.set_seed(a.seed)
    dev = C.DEVICE
    X = np.load(dom_path(a.dataset, a.domain))
    need = MIN_INPUT.get(a.backbone, 8)
    if X.shape[1] < need:
        print("SKIP %s: %s needs >=%dpx, domain is %dpx"
              % (tag, a.backbone, need, X.shape[1]))
        return
    tr = GPUBatcher(X, labels, sp["train"], bs, True, dev)
    model = build_cls(a.backbone, ncls, pretrained=True).to(dev)
    if a.backbone.startswith("convnext"):
        peak = a.lr if a.lr != 0.01 else 3e-4
        opt = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=0.05)
    else:
        peak = a.lr
        opt = torch.optim.SGD(model.parameters(), lr=peak, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=peak, epochs=a.epochs,
                                              steps_per_epoch=len(tr), pct_start=0.25)
    scaler = torch.cuda.amp.GradScaler()
    rng = np.random.RandomState(a.seed)
    t0 = time.time(); best = -1; best_state = None
    for ep in range(1, a.epochs + 1):
        model.train()
        for xb, yb in tr.epoch(rng):
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        if ep % 5 == 0 or ep == a.epochs:
            lg = logits_on(model, X, sp["val"], dev)
            acc = float((lg.argmax(1) == np.asarray(labels)[sp["val"]]).mean())
            if acc > best:
                best = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ep{ep:3d} val {acc:.4f} [{(time.time()-t0)/60:.1f}m]", flush=True)
    model.load_state_dict(best_state)

    doms = all_domains(a.dataset) if a.eval_domains == "all" else a.eval_domains.split(",")
    res = {"classes": np.array(sp["classes"]), "val_acc_matched": best,
           "y_test": np.asarray(labels)[sp["test"]], "y_val": np.asarray(labels)[sp["val"]],
           "domains": np.array(doms)}
    base_shape = X.shape[1]
    for d in doms:
        Y = np.load(dom_path(a.dataset, d), mmap_mode="r")
        if Y.shape[1] < need:
            del Y
            continue
        if Y.shape[1] != base_shape and not (d.startswith("lr_") or a.domain.startswith("lr_")):
            continue
        try:
            res[f"test_{d}"] = logits_on(model, Y, sp["test"], dev)
            res[f"val_{d}"] = logits_on(model, Y, sp["val"], dev)
        except Exception as e:
            print(f"  skip eval on {d}: {type(e).__name__}", flush=True)
        del Y
    np.savez_compressed(dst, **res)
    accs = {d: float((res[f"test_{d}"].argmax(1) == res["y_test"]).mean())
            for d in doms if f"test_{d}" in res}
    print(f"[{tag}] {(time.time()-t0)/60:.1f}m " + json.dumps({k: round(v, 4) for k, v in accs.items()}))


if __name__ == "__main__":
    main()
