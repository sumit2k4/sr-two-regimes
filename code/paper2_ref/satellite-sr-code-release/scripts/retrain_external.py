"""Retrain an external (recent) method under OUR common protocol for Table 11.

This is a real adapter (not a re-implementation). For one method and one seed it:
  1. runs the official repo's TRAIN command under our fixed UC Merced x4 split;
  2. runs the official repo's INFER/TEST command to write super-resolved (SR) images
     for our test split into runs/<method>_seed<seed>/sr/;
  3. scores those SR images against our HR test images with OUR metric scripts
     (scripts/score_folder.py), writing runs/<method>_seed<seed>/eval.txt.

Because each official repo has its own CLI, fill TRAIN_CMDS and INFER_CMDS below once you
know each repo's commands. Placeholders available in the command strings:
  {split}    -> our training split file      (e.g. splits/ucmerced_train.txt)
  {testlist} -> our test split file          (e.g. splits/ucmerced_test.txt)
  {seed}     -> the random seed
  {ckpt}     -> runs/<method>_seed<seed>/model.pth   (where the repo should save/load)
  {srdir}    -> runs/<method>_seed<seed>/sr          (where the repo should write SR PNGs)

Until a method's commands are filled, this script exits with a clear message and writes
NO numbers -- the Table 11 cell stays 'retrain'.
"""
import argparse
import os
import subprocess

# ---- Fill these per method (copy from each official repo's README) -------------------
# Example for an EDSR-style repo:
#   TRAIN_CMDS["dtwstsr"] = (
#     "python external/dtwstsr/main.py --train_list {split} --scale 4 "
#     "--epochs 200 --seed {seed} --save {ckpt}")
#   INFER_CMDS["dtwstsr"] = (
#     "python external/dtwstsr/test.py --test_list {testlist} --scale 4 "
#     "--resume {ckpt} --output_dir {srdir}")
TRAIN_CMDS = {
    "bd_vitgan": None,
    "dtwstsr": None,
    "ham": None,
}
INFER_CMDS = {
    "bd_vitgan": None,
    "dtwstsr": None,
    "ham": None,
}
# --------------------------------------------------------------------------------------


def run(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="splits/ucmerced_train.txt")
    ap.add_argument("--testlist", default="splits/ucmerced_test.txt")
    ap.add_argument("--hr-dir", default=None,
                    help="Directory of HR test images (defaults to paths inside testlist).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_cmd = TRAIN_CMDS.get(args.method)
    infer_cmd = INFER_CMDS.get(args.method)
    if not train_cmd or not infer_cmd:
        raise SystemExit(
            f"[{args.method}] TRAIN_CMDS/INFER_CMDS not filled in scripts/retrain_external.py. "
            f"Add the official repo's train and inference commands (use the {{split}}, "
            f"{{testlist}}, {{seed}}, {{ckpt}}, {{srdir}} placeholders). No number is written; "
            f"the Table 11 cell stays 'retrain'.")

    out_dir = os.path.join("runs", f"{args.method}_seed{args.seed}")
    sr_dir = os.path.join(out_dir, "sr")
    ckpt = os.path.join(out_dir, "model.pth")
    os.makedirs(sr_dir, exist_ok=True)

    fmt = dict(split=args.split, testlist=args.testlist, seed=args.seed,
               ckpt=ckpt, srdir=sr_dir)

    print(f"[{args.method}] seed {args.seed}: training under our protocol ...")
    run(train_cmd.format(**fmt))
    print(f"[{args.method}] seed {args.seed}: inference on our test split ...")
    run(infer_cmd.format(**fmt))

    # score the produced SR images with OUR metrics
    print(f"[{args.method}] seed {args.seed}: scoring with our metric scripts ...")
    score = (f"python scripts/score_folder.py --sr-dir {sr_dir} "
             f"--testlist {args.testlist} --out {os.path.join(out_dir, 'eval.txt')}")
    if args.hr_dir:
        score += f" --hr-dir {args.hr_dir}"
    run(score)
    print(f"[{args.method}] seed {args.seed}: wrote {os.path.join(out_dir, 'eval.txt')}")


if __name__ == "__main__":
    main()
