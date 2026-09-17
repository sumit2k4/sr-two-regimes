"""Per-image reconstruction quality (PSNR / SSIM / LPIPS) of every SR domain
against the HR reference, on the validation and test splits.

These per-image values are what the correlation analysis in Section V uses to
ask whether pixel fidelity or perceptual fidelity predicts the downstream
classification gain.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CACHE = os.path.join(C.RUNS, "cache")


def domains(ds):
    pre = ds + "_"
    out = []
    for f in sorted(os.listdir(CACHE)):
        if f.startswith(pre) and f.endswith(".npy"):
            d = f[len(pre):-4]
            if d != "hr" and not d.startswith("lr_"):
                out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    import lpips
    sp = C.get_splits(a.dataset)
    HR = np.load(os.path.join(CACHE, a.dataset + "_hr.npy"), mmap_mode="r")
    net = lpips.LPIPS(net="alex").to(C.DEVICE)
    out = {}
    for d in domains(a.dataset):
        X = np.load(os.path.join(CACHE, a.dataset + "_" + d + ".npy"), mmap_mode="r")
        for split in ["val", "test"]:
            idx = sp[split]
            ps, ss, lp = [], [], []
            for k in range(0, len(idx), 16):
                ids = np.asarray(idx[k:k + 16])
                xs = np.ascontiguousarray(X[ids]).astype(np.float32) / 255.0
                hs = np.ascontiguousarray(HR[ids]).astype(np.float32) / 255.0
                for i in range(len(ids)):
                    ps.append(C.psnr(xs[i], hs[i]))
                    ss.append(C.ssim(xs[i], hs[i]))
                tx = torch.from_numpy(xs).permute(0, 3, 1, 2).to(C.DEVICE) * 2 - 1
                th = torch.from_numpy(hs).permute(0, 3, 1, 2).to(C.DEVICE) * 2 - 1
                with torch.no_grad():
                    lp.append(net(tx, th).flatten().cpu().numpy())
            out[d + "_" + split + "_psnr"] = np.array(ps, np.float32)
            out[d + "_" + split + "_ssim"] = np.array(ss, np.float32)
            out[d + "_" + split + "_lpips"] = np.concatenate(lp).astype(np.float32)
            print(d, split, "PSNR %.3f SSIM %.4f LPIPS %.4f"
                  % (np.mean(ps), np.mean(ss), np.mean(np.concatenate(lp))), flush=True)
        del X
    np.savez_compressed(os.path.join(C.RESULTS, "srquality_" + a.dataset + ".npz"), **out)
    summ = {}
    for d in domains(a.dataset):
        summ[d] = {m: [float(np.mean(out[d + "_test_" + m])), float(np.std(out[d + "_test_" + m]))]
                   for m in ["psnr", "ssim", "lpips"]}
    json.dump(summ, open(os.path.join(C.RESULTS, "srquality_" + a.dataset + ".json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
