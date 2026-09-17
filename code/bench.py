"""Measure parameters, multiply-accumulate cost and GPU latency of every stage,
so the deployment cost of the gated ensemble is reported honestly."""
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sr_models import build_sr, tiled_sr, n_params
from cls_models import build_cls
from qge import Gate


def gflops(model, x):
    """Convolution/linear MACs by forward hooks, reported as GFLOPs (2 x MACs)."""
    total = [0]
    hs = []

    def hk(m, i, o):
        if isinstance(m, torch.nn.Conv2d):
            total[0] += m.weight.numel() * o.shape[-1] * o.shape[-2] / max(1, m.groups) * m.groups
        elif isinstance(m, torch.nn.ConvTranspose2d):
            total[0] += m.weight.numel() * o.shape[-1] * o.shape[-2]
        elif isinstance(m, torch.nn.Linear):
            total[0] += m.weight.numel() * (o.numel() // o.shape[-1])
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            hs.append(m.register_forward_hook(hk))
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return 2 * total[0] / 1e9


def timeit(fn, x, n=30):
    for _ in range(5):
        fn(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn(x)
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000.0


def main(ds="ucmerced"):
    dev = C.DEVICE
    cfg = C.DATASETS[ds]
    sp = C.get_splits(ds)
    out = {}
    lr = torch.rand(1, 3, cfg["lr"], cfg["lr"], device=dev)
    hr = torch.rand(1, 3, cfg["hr"], cfg["hr"], device=dev)
    for m in ["srcnn", "edsr", "hfgan_p", "hfgan_g"]:
        ck_p = os.path.join(C.RUNS, "sr", ds + "_" + m + ".pt")
        if not os.path.exists(ck_p):
            continue
        ck = torch.load(ck_p, map_location="cpu")
        net = build_sr(m, C.SCALE, img_size=ck["p_lr"]).to(dev).eval()
        net.load_state_dict(ck["model"])
        tile = ck["p_lr"]
        need = m.startswith("hfgan") and cfg["lr"] > tile

        def f(x, net=net, need=need, tile=tile):
            with torch.no_grad():
                return tiled_sr(net, x, tile=tile) if need else net(x)
        out[m] = dict(label=m.upper().replace("_", "-"),
                      params_m=n_params(net) / 1e6,
                      gflops=gflops(net, lr[:, :, :tile, :tile]) * (
                          (cfg["lr"] / tile) ** 2 if need else 1.0),
                      ms=timeit(f, lr))
    cls = build_cls("resnet18", len(sp["classes"])).to(dev).eval()

    def fc(x):
        with torch.no_grad():
            return cls(x)
    out["resnet18"] = dict(label="ResNet-18 branch", params_m=n_params(cls) / 1e6,
                           gflops=gflops(cls, hr), ms=timeit(fc, hr))
    g = Gate(20, 3).to(dev).eval()
    z = torch.rand(1, 20, device=dev)

    def fg(x):
        with torch.no_grad():
            return g(x)
    out["gate"] = dict(label="Gate MLP", params_m=n_params(g) / 1e6,
                       gflops=gflops(g, z), ms=timeit(fg, z))
    tot_ms = out["gate"]["ms"] + 3 * out["resnet18"]["ms"]
    for m in ["edsr", "hfgan_g"]:
        if m in out:
            tot_ms += out[m]["ms"]
    out["qge_total"] = dict(label="\\textbf{QGE end to end}",
                            params_m=sum(out[k]["params_m"] for k in
                                         ["edsr", "hfgan_g", "gate"] if k in out)
                            + 3 * out["resnet18"]["params_m"],
                            gflops=sum(out[k]["gflops"] for k in
                                       ["edsr", "hfgan_g", "gate"] if k in out)
                            + 3 * out["resnet18"]["gflops"],
                            ms=tot_ms)
    json.dump(out, open(os.path.join(C.RESULTS, "efficiency.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ucmerced")
