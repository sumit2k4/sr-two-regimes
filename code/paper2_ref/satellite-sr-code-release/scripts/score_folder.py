"""Score a folder of super-resolved (SR) images against HR test images.

External methods output SR image files rather than checkpoints loadable into our
Generator, so this scores those files with OUR metric scripts (src/metrics.py). This
guarantees that every method in Table 11 -- ours, baselines and retrained recent methods
-- is evaluated with an identical PSNR/SSIM/LPIPS implementation.

Matching rule: for each HR image listed in --testlist, the SR image is the file in
--sr-dir with the SAME basename (any image extension). HR and SR are compared at the HR
resolution; if an SR image differs in size it is bicubic-resized to the HR size first.

Usage:
    python scripts/score_folder.py --sr-dir runs/dtwstsr_seed42/sr \
        --testlist splits/ucmerced_test.txt --out runs/dtwstsr_seed42/eval.txt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# allow running as `python scripts/score_folder.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import metrics

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _to_tensor_pm1(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # [-1,1], same convention as the dataset
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def _find_sr(sr_dir, stem):
    for ext in IMG_EXT:
        p = Path(sr_dir) / f"{stem}{ext}"
        if p.exists():
            return p
    # fall back: any file starting with the stem (some repos append _x4 etc.)
    cands = [p for p in Path(sr_dir).iterdir()
             if p.stem.startswith(stem) and p.suffix.lower() in IMG_EXT]
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr-dir", required=True)
    ap.add_argument("--testlist", required=True)
    ap.add_argument("--hr-dir", default=None,
                    help="Override HR directory; default uses the paths inside testlist.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hr_paths = [ln.strip() for ln in Path(args.testlist).read_text().splitlines() if ln.strip()]

    ps, ss, lp, missing = [], [], [], 0
    for hr_path in hr_paths:
        stem = Path(hr_path).stem
        hr_full = (Path(args.hr_dir) / Path(hr_path).name) if args.hr_dir else Path(hr_path)
        sr_path = _find_sr(args.sr_dir, stem)
        if sr_path is None or not Path(hr_full).exists():
            missing += 1
            continue
        hr_img = Image.open(hr_full).convert("RGB")
        sr_img = Image.open(sr_path).convert("RGB")
        if sr_img.size != hr_img.size:
            sr_img = sr_img.resize(hr_img.size, Image.BICUBIC)
        hr_t, sr_t = _to_tensor_pm1(hr_img), _to_tensor_pm1(sr_img)
        ps.append(metrics.psnr(sr_t, hr_t))
        ss.append(metrics.ssim(sr_t, hr_t))
        if not args.no_lpips:
            lp.append(metrics.lpips(sr_t, hr_t, device))

    if not ps:
        raise SystemExit(f"No SR/HR pairs matched (missing={missing}). "
                         f"Check --sr-dir filenames match HR basenames.")
    psnr_v, ssim_v = float(np.mean(ps)), float(np.mean(ss))
    lpips_v = float(np.mean(lp)) if lp else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"PSNR {psnr_v:.4f}\nSSIM {ssim_v:.4f}\n")
        if lpips_v is not None:
            f.write(f"LPIPS {lpips_v:.4f}\n")
    print(f"scored {len(ps)} images (missing {missing}) -> "
          f"PSNR {psnr_v:.3f}  SSIM {ssim_v:.4f}"
          + (f"  LPIPS {lpips_v:.4f}" if lpips_v is not None else ""))


if __name__ == "__main__":
    main()
