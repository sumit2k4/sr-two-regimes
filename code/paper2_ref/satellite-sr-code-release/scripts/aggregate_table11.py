"""Build Table 11 (recent-methods comparison on UC Merced x4).

Two kinds of rows, kept strictly separate:
  1. Published figures (MBGPIN, DEGAN) -> taken verbatim from published_figures.yaml,
     printed with a trailing '*'. Never produced or overwritten here.
  2. Retrained methods (BD-VITGAN, DTWSTSR, HAM) -> mean +/- std across seeds, read from
     runs/<method>_seed<seed>/eval.txt. If a result file is missing the cell is written
     as 'retrain' (placeholder) -- no number is invented.

Usage:
    python scripts/aggregate_table11.py \
        --published configs/recent/published_figures.yaml \
        --runs runs --out runs/table11.csv
"""
import argparse
import csv
import statistics
from pathlib import Path

import yaml


def _fmt(mean, std):
    return f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"


def _load_retrained(runs_root, method, seeds=(42, 123, 999)):
    ps, ss, lp = [], [], []
    have_lpips = True
    for s in seeds:
        f = Path(runs_root) / f"{method}_seed{s}" / "eval.txt"
        if not f.exists():
            return None
        d = dict(line.split() for line in f.read_text().split("\n") if " " in line)
        ps.append(float(d["PSNR"])); ss.append(float(d["SSIM"]))
        if "LPIPS" in d:
            lp.append(float(d["LPIPS"]))
        else:
            have_lpips = False
    g = lambda v: (statistics.mean(v), statistics.pstdev(v) if len(v) > 1 else None)
    out = {"psnr": g(ps), "ssim": g(ss)}
    out["lpips"] = g(lp) if (have_lpips and lp) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", default="configs/recent/published_figures.yaml")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/table11.csv")
    args = ap.parse_args()

    pub = yaml.safe_load(Path(args.published).read_text())
    rows = []

    for name in ("MBGPIN", "DEGAN"):
        e = pub[name]
        rows.append([name, "published",
                     f"{e['psnr']:.2f}*" if e.get("psnr") else "n/r",
                     f"{e['ssim']:.3f}*" if e.get("ssim") else "n/r",
                     f"{e['lpips']:.3f}*" if e.get("lpips") else "n/r"])

    for name in ("bd_vitgan", "dtwstsr", "ham"):
        r = _load_retrained(args.runs, name)
        if r is None:
            rows.append([name, "retrain", "retrain", "retrain", "retrain"])
        else:
            lpips_cell = _fmt(*r["lpips"]) if r["lpips"] else "n/r"
            rows.append([name, "retrain", _fmt(*r["psnr"]), _fmt(*r["ssim"]), lpips_cell])

    # the proposed model row (from its own evaluated runs, if present)
    prop = _load_retrained(args.runs, "ucmerced_proposed")
    if prop:
        lpips_cell = _fmt(*prop["lpips"]) if prop["lpips"] else "n/r"
        rows.append(["Proposed", "ours", _fmt(*prop["psnr"]), _fmt(*prop["ssim"]), lpips_cell])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Type", "PSNR", "SSIM", "LPIPS"])
        w.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows). "
          f"Published values carry '*'; unmeasured cells say 'retrain' (never fabricated).")


if __name__ == "__main__":
    main()
