# Next-Generation Deep Hybrid Fusion GAN with Transformer Attention for Remote Sensing Image Super-Resolution

Official implementation accompanying the IJIES manuscript (Paper ID 20263246).
This repository is released to support full reproducibility and transparency of
the reported results.

> Replace the placeholder repository URL in the manuscript with the actual public
> URL once the repository is published, e.g.
> `https://github.com/<author-account>/satellite-sr-code-release`.

## 1. Method overview

The generator follows a sequentially decoupled local→global design:

```
LR image
  → 3×3 shallow convolution (3→64)
  → 16 × RRDB              (nf=64, gc=32, residual scaling 0.2)
  → Transformer encoder    (4 layers, 8 heads, embed_dim 256, MLP ratio 4, ReLU)
  → Self-Attention Refinement Block (SARB: channel reduction 4, 7×7 spatial conv)
  → Hybrid Feature Fusion  (concat[RRDB, SARB] → 3×3 conv 128→64)
  → 2 × PixelShuffle (×2 each → overall ×4)
  → 3×3 reconstruction conv + Tanh
SR image
```

The discriminator is an **eight-layer global convolutional classifier**
(channels 64→64→128→128→256→256→512→512, stride-2 on alternate layers,
BatchNorm on all layers except the first, LeakyReLU 0.2, then Flatten + two FC
layers → single logit). It is trained with a binary cross-entropy (with-logits)
adversarial objective. *No spectral normalization is used.*

### Loss
`L_G = 1.0 · L1 + 0.01 · VGG19(relu5_4) + 0.005 · BCEWithLogits`  (manuscript Eq. 26)

## 2. Installation

```bash
git clone https://github.com/<author-account>/satellite-sr-code-release.git
cd satellite-sr-code-release
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Print the exact parameter counts and FLOPs:

```bash
python -m src.models          # prints generator / discriminator parameter counts
```

## 3. Datasets

Download the four benchmarks and point `make_splits.py` at each image directory:

| Dataset | Images | Classes | Tile size |
|---------|--------|---------|-----------|
| UC Merced Land Use | 2,100 | 21 | 256×256 |
| NWPU-RESISC45 | 31,500 | 45 | 256×256 |
| RSSCN7 | 2,800 | 7 | 400×400 |
| AID | 10,000 | 30 | 600×600 |

## 4. Reproducing every table

**Step 1 — fixed 70/15/15 splits (image-level, seed 42):**

```bash
python scripts/make_splits.py --image-dir /data/UCMerced     --name ucmerced --seed 42
python scripts/make_splits.py --image-dir /data/NWPU-RESISC45 --name nwpu     --seed 42
python scripts/make_splits.py --image-dir /data/AID          --name aid      --seed 42
python scripts/make_splits.py --image-dir /data/RSSCN7       --name rsscn7   --seed 42
```

The generated split files are committed under `splits/` so the exact partition
is fixed across machines. The committed UC Merced lists (`splits/ucmerced_{train,val,test}.txt`,
1470 / 315 / 315 images for the seed-42 70/15/15 partition) store **image basenames**;
`make_splits.py` writes absolute paths on your machine, so either re-run it on your copy of
UC Merced or prepend your image directory to the committed basenames.

**Step 2 — train (three seeds → mean ± std):**

```bash
for SEED in 42 123 999; do
  python train.py --config configs/default.yaml --seed $SEED \
      --train-split splits/ucmerced_train.txt \
      --val-split   splits/ucmerced_val.txt \
      --out runs/ucmerced_seed$SEED
done
```

The checkpoint with the **best validation PSNR** is saved as `generator_best.pth`.

**Step 3 — evaluate (Tables 5–8):**

```bash
python evaluate.py --config configs/default.yaml \
    --ckpt runs/ucmerced_seed42/generator_best.pth \
    --test-split splits/ucmerced_test.txt
```

**Step 4 — non-ideal degradation (Table 10):** set `degradation: blur_noise`
in the config (7×7 Gaussian blur σ=1.2 → ×4 bicubic → additive noise σ=5/255),
then repeat Steps 2–3.

**Ablation study (Table 9)** — one command per variant via `--variant`:

```bash
bash scripts/reproduce_table9_ablation.sh
# or a single variant:
python train.py --config configs/default.yaml --variant I \
    --train-split splits/ucmerced_train.txt --val-split splits/ucmerced_val.txt \
    --out runs/ablation_I_seed42 --seed 42
```

Variants: `I` = CNN-only (L1, no GAN), `II` = CNN+Transformer (no GAN),
`III` = CNN+GAN (no Transformer), `IV` = full proposed model. Variant IV gives the
highest PSNR/SSIM (highest pixel consistency); Variant I the highest LPIPS.

**Main benchmarks (Tables 5–8):**

```bash
bash scripts/reproduce_table5_8.sh
```

**Recent-methods comparison (Table 11)** — UC Merced ×4. The table reported in the
paper has two kinds of rows, kept strictly separate:

1. **Published figures** — MBGPIN [29] (31.34 / 0.912) and DEGAN [31] (28.90 / 0.796),
   quoted with a `*` and an explicit citation in
   `configs/recent/published_figures.yaml`. They follow each paper's own protocol, so
   they are indicative, not strictly protocol-identical.
2. **Protocol-identical rows** — the **Proposed** model and the **SwinIR** baseline are
   trained and scored with *this* repo's code under the common pipeline (×4 bicubic,
   fixed 70/15/15 split, identical PSNR/SSIM/LPIPS scripts), so their three columns are
   directly comparable. Reproduce them with:

```bash
bash scripts/reproduce_table11.sh        # trains/evaluates Proposed; writes runs/table11.csv
bash scripts/reproduce_table5_8.sh       # provides the SwinIR baseline row, same protocol
```

Other recent Transformer-/GAN-/wavelet-/Mamba-based methods (BD-VITGAN [32],
DTWSTSR [34], HAM [40]) report only on different benchmarks/protocols and are compared
**qualitatively** in Table 1 of the paper. They are therefore *not* listed as numeric
rows in Table 11. If you want to add a strictly protocol-identical row for any of them,
the optional adapter retrains the official repo under our pipeline:

```bash
# optional extension: fill official_repo URLs in configs/recent/*.yaml and the
# TRAIN_CMDS/INFER_CMDS dicts in scripts/retrain_external.py, then re-run:
bash scripts/reproduce_table11.sh
```

No benchmark value is fabricated: only the two published figures are quoted (with `*`),
every other listed row is measured by this repo, and any method you have not retrained
is simply omitted rather than given an invented number.

## 5. Baselines

Bicubic, EDSR, ESRGAN and SwinIR are retrained **from scratch** (no pretrained
checkpoints) under the identical degradation, splits, augmentation, optimizer,
learning-rate schedule, batch size and number of epochs, using their official
PyTorch implementations. Configuration files for each baseline are provided in
`configs/` (add `configs/baseline_*.yaml` as needed) and the exact commands are
listed in this README so that every comparison number can be reproduced.

## 6. Training configuration (summary)

| Setting | Value |
|---------|-------|
| Degradation | ×4 bicubic (or blur+noise for Table 10) |
| HR / LR patch | 256×256 / 64×64 |
| Augmentation | horizontal flip, vertical flip, 90° rotation |
| Optimizer | Adam (β1=0.9, β2=0.999) |
| Learning rate | 1e-4, ×0.5 every 50 epochs |
| Epochs / batch | 200 / 16 |
| Seeds | 42, 123, 999 |
| Hardware (paper) | NVIDIA RTX 3060 GPU |

## 7. Citation

```bibtex
@article{prithviraj_ijies_hybridgan,
  title   = {Next-Generation Deep Hybrid Fusion GAN with Transformer Attention
             for Remote Sensing Image Super-Resolution},
  author  = {Prithviraj and Rajesh, I. S. and Bharathi Malakreddy, A.},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2025}
}
```

## License
Released under the MIT License (see `LICENSE`).
