# How to get REAL numbers for Table 11 (BD-VITGAN, DTWSTSR, HAM)

Two rows in Table 11 are already real and need no run:
* **MBGPIN** and **DEGAN** — published UC Merced ×4 figures (quoted with `*`).

Three rows must be measured on your GPU: **BD-VITGAN, DTWSTSR, HAM**. Follow these
steps. Total work is mostly waiting on training; the wiring is done.

---

## Prerequisites
* A CUDA GPU (these methods won't finish in reasonable time on CPU).
* The UC Merced HR images downloaded, and your fixed splits created:
  ```bash
  python scripts/make_splits.py --image-dir /path/to/UCMerced --name ucmerced --seed 42
  ```
  This writes `splits/ucmerced_{train,val,test}.txt`.
* `pip install -r requirements.txt`

## Step 1 — Put the official repo URL in each config
Edit these three files and replace `<fill official ... repo URL>`:
* `configs/recent/bd_vitgan.yaml`
* `configs/recent/dtwstsr.yaml`
* `configs/recent/ham.yaml`

Find the URL in each paper (usually a GitHub link in the abstract or a "Code
availability" line).

## Step 2 — Tell the adapter how to launch each repo
Open `scripts/retrain_external.py` and fill **two** dicts, `TRAIN_CMDS` and
`INFER_CMDS`, one line per method. Copy the train/test commands from each repo's own
README and substitute these placeholders:

| placeholder | becomes |
|-------------|---------|
| `{split}`    | `splits/ucmerced_train.txt` (your training list) |
| `{testlist}` | `splits/ucmerced_test.txt`  (your test list) |
| `{seed}`     | 42 / 123 / 999 |
| `{ckpt}`     | `runs/<method>_seed<seed>/model.pth` (where the repo saves/loads) |
| `{srdir}`    | `runs/<method>_seed<seed>/sr` (where the repo must write SR PNGs) |

Example (an EDSR-style repo — adapt to the real CLI):
```python
TRAIN_CMDS["dtwstsr"] = (
    "python external/dtwstsr/main.py --train_list {split} --scale 4 "
    "--epochs 200 --seed {seed} --save {ckpt}")
INFER_CMDS["dtwstsr"] = (
    "python external/dtwstsr/test.py --test_list {testlist} --scale 4 "
    "--resume {ckpt} --output_dir {srdir}")
```
The only hard requirement: **inference must write one SR image per test image into
`{srdir}`, named with the same basename as the HR image.** If a repo can only take a
folder of LR inputs, first generate the LR test images with `make_degradation`-style
bicubic ×4 (see `src/degrade.py`) and point the repo at that folder.

## Step 3 — Run everything
```bash
bash scripts/reproduce_table11.sh
```
This will, for each of the three methods and seeds 42/123/999:
1. `git clone` the repo into `external/<method>/`,
2. train it on your UC Merced split,
3. run its inference to produce SR images for your test split,
4. **score those SR images with your own PSNR/SSIM/LPIPS** (`scripts/score_folder.py`),
   writing `runs/<method>_seed<seed>/eval.txt`,
5. aggregate everything into `runs/table11.csv`.

It also trains+evaluates your own model on UC Merced (the "Proposed" row) if you
haven't already.

## Step 4 — Read the result and update the paper
```bash
cat runs/table11.csv
```
You'll get mean ± std for each retrained method. Copy those into the manuscript's
Table 11, replacing the `retrain†` cells. Leave MBGPIN/DEGAN with their `*` published
values.

---

## Sanity checks / troubleshooting
* **"No SR/HR pairs matched"** — the SR filenames don't match HR basenames. Either
  rename the repo's outputs or adjust the `_find_sr` fallback in `score_folder.py`.
* **Different image size** — `score_folder.py` bicubic-resizes SR to the HR size before
  scoring, so minor size mismatches are handled, but a wrong scale factor will tank PSNR.
* **One method won't train** (dependency hell) — you can drop it: remove it from the
  `RETRAIN=(...)` list in `reproduce_table11.sh` and from Table 11, and note in the paper
  that it could not be retrained under the common protocol. Two retrained methods plus
  two published figures is already a solid recent-methods comparison.
* **No GPU for all three** — run whichever you can; the others stay as `retrain` and you
  trim them from the table.

## Integrity reminder
`score_folder.py` is the single metric implementation used for **every** method, so the
retrained numbers are directly comparable to your own. Nothing is fabricated: a method you
don't run stays as a `retrain` placeholder until you measure it.
