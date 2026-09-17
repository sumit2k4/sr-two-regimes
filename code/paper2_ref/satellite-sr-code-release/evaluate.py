"""Evaluation entry point: reports PSNR / SSIM / LPIPS on a test split.

Writes the metrics both to stdout and to <out_dir>/eval.txt (key value per line) so the
Table 11 aggregator can read per-run results.

Example:
    python evaluate.py --config configs/default.yaml \
        --ckpt runs/ucmerced_proposed_seed42/generator_best.pth \
        --test-split splits/ucmerced_test.txt --variant IV
"""
import argparse
import os
import torch

from src.utils import load_config
from src.models import Generator
from src.dataset import build_loader
from src import metrics


VARIANT_FLAGS = {
    "I":   dict(use_transformer=False, use_sarb=False, use_hff=False),
    "II":  dict(use_transformer=True,  use_sarb=True,  use_hff=True),
    "III": dict(use_transformer=False, use_sarb=False, use_hff=False),
    "IV":  dict(use_transformer=True,  use_sarb=True,  use_hff=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-split", required=True)
    ap.add_argument("--variant", choices=["I", "II", "III", "IV"], default="IV",
                    help="Must match the variant the checkpoint was trained with.")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flags = VARIANT_FLAGS[args.variant]

    netG = Generator(num_blocks=cfg["model"]["rrdb_blocks"],
                     embed_dim=cfg["model"]["embed_dim"],
                     num_heads=cfg["model"]["num_heads"],
                     num_layers=cfg["model"]["num_layers"],
                     use_transformer=flags["use_transformer"],
                     use_sarb=flags["use_sarb"],
                     use_hff=flags["use_hff"]).to(device)
    netG.load_state_dict(torch.load(args.ckpt, map_location=device))
    netG.eval()

    loader = build_loader(args.test_split, 1, cfg["data"]["hr_size"], cfg["data"]["scale"],
                          cfg["data"]["degradation"], augment=False, shuffle=False)
    ps, ss, lp = [], [], []
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = netG(lr)
            ps.append(metrics.psnr(sr, hr))
            ss.append(metrics.ssim(sr, hr))
            if not args.no_lpips:
                lp.append(metrics.lpips(sr, hr, device))
    n = len(ps)
    psnr_v, ssim_v = sum(ps) / n, sum(ss) / n
    lpips_v = (sum(lp) / n) if lp else None
    print(f"PSNR : {psnr_v:.3f} dB")
    print(f"SSIM : {ssim_v:.4f}")
    if lpips_v is not None:
        print(f"LPIPS: {lpips_v:.4f}")

    # write eval.txt next to the checkpoint for the aggregator
    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    with open(os.path.join(out_dir, "eval.txt"), "w") as f:
        f.write(f"PSNR {psnr_v:.4f}\nSSIM {ssim_v:.4f}\n")
        if lpips_v is not None:
            f.write(f"LPIPS {lpips_v:.4f}\n")


if __name__ == "__main__":
    main()
