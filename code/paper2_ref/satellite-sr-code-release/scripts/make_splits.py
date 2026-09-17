"""Create fixed 70/15/15 train/val/test split files for a dataset.

Splitting is performed at the IMAGE level (before any patch extraction or
augmentation) using a fixed seed, so there is no cross-split contamination.

Example:
    python scripts/make_splits.py --image-dir /data/UCMerced --name ucmerced --seed 42
"""
import os
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--ratios", default="0.7,0.15,0.15")
    args = ap.parse_args()

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files = []
    for root, _, names in os.walk(args.image_dir):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    files.sort()
    random.Random(args.seed).shuffle(files)

    r_tr, r_va, _ = (float(x) for x in args.ratios.split(","))
    n = len(files)
    n_tr, n_va = int(n * r_tr), int(n * r_va)
    parts = {"train": files[:n_tr], "val": files[n_tr:n_tr + n_va], "test": files[n_tr + n_va:]}

    os.makedirs(args.out_dir, exist_ok=True)
    for split, items in parts.items():
        path = os.path.join(args.out_dir, f"{args.name}_{split}.txt")
        with open(path, "w") as f:
            f.write("\n".join(items) + "\n")
        print(f"{path}: {len(items)} images")


if __name__ == "__main__":
    main()
