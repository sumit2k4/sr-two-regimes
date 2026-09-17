# Controlled evaluation of super-resolution for land-cover classification

This repository holds the experimental code for a controlled study of
super-resolution as a preprocessing step for remote sensing scene
classification. The protocol separates two things that are often conflated:
reconstructed detail from classifier input size, by adding an interpolated
control at matched spatial size; and two deployment regimes, one in which the
classifier is retrained on the reconstructions and one in which an existing
high-resolution-trained classifier is reused unchanged.

The pipeline covers dataset preparation, two degradation protocols, four
super-resolution models spanning the perception-distortion trade-off, a full
classification factorial across two backbones and three seeds with paired
significance testing, and an adaptive per-image fusion gate with an oracle
headroom analysis.

Findings are reported in the associated article, which is in preparation and not
included here.

## What is in this repository

```
code/          every script: data, degradation, SR models, classifiers, gate, analysis
results/       json artefacts, per-image quality arrays, fixed split files
sr_summaries/  per-model training logs and validation PSNR summaries
```

The manuscript, its LaTeX sources and its figures are **not** included while the
article is under preparation. `code/build_paper.sh` regenerates all tables,
number macros and figures from `results/`, so the analysis is fully auditable
without them.

Deliberately **not** committed, because of size:

| Excluded | Size | How to regenerate |
|---|---|---|
| `data/` | 1.1 GB | `python code/fetch_data.py` |
| `runs/cache/` | 8.2 GB | `sh code/run_stage2.sh` |
| `runs/cls/` | 115 MB | same |
| `runs/sr/*.pt` | 286 MB | `sh code/run_sr_all.sh` |

The fixed split files in `results/` are committed, so a rerun reproduces exactly
the same partitions.

## Reproducing

```sh
python code/fetch_data.py       # UC Merced and EuroSAT from the HuggingFace mirrors
sh     code/run_sr_all.sh       # 8 super-resolution trainings
sh     code/run_stage2.sh       # domain caches, quality scoring, classifiers
sh     code/run_convnext.sh     # the ConvNeXt-Tiny factorial and the gate
python code/bench.py ucmerced   # parameters, GFLOPs, latency
sh     code/build_paper.sh      # all tables, macros and figures
```

End to end this is roughly 14 hours on a single 8 GB GPU.

## Protocol, frozen

* scale factor 4; stratified 70/15/15 split per dataset; split seed 42
* the degradation is applied **after** splitting, and every SR model is trained
  only on the training split, so no test scene reaches any SR network
* every domain except the native LR one is presented at the same spatial size,
  which is what removes the input-size confound
* classifier seeds 42, 123, 999; all results are mean and standard deviation
* the gate is fitted on the validation split only, with the branches frozen
* P1 = bicubic decimation; P2 = Gaussian blur (7x7, sigma 1.2) -> bicubic
  decimation -> additive noise (sigma 5/255) -> JPEG q85. No SR model sees P2.

## Scripts

| file | role |
|---|---|
| `common.py` | splits, PSNR/SSIM, protocol constants |
| `degrade.py` | the two degradation pipelines |
| `sr_models.py` | SRCNN, EDSR, the HFGAN generator and discriminator, tiled inference |
| `train_sr.py` | trains one SR model on one dataset |
| `make_sr_cache.py` | materialises every input domain as a uint8 `.npy` |
| `sr_eval.py` | per-image PSNR / SSIM / LPIPS against the HR reference |
| `cls_models.py`, `train_cls.py` | branch classifiers and the full evaluation matrix |
| `qge.py`, `qge_ext.py` | reliability descriptor, gate, all fusion baselines, ablations |
| `gate_diag.py` | the per-image oracle bound that explains the negative result |
| `analyze.py`, `analyze_transfer.py` | matched-regime and transfer-regime tables |
| `make_tables*.py`, `figures*.py` | tables, number macros and all figures |
| `bench.py` | parameters, GFLOPs, measured latency |

`code/paper2_ref/` contains the previously released super-resolution code, from
which the HFGAN generator and discriminator are imported verbatim so that the
reconstruction stage is exactly the previously published architecture.

## Hardware

All experiments ran on a single NVIDIA RTX 2070 Super (8 GB). The HFGAN
generator carries absolute positional codes for a fixed token grid, so training
uses LR patches at that token count and inference on larger frames uses
overlap-and-blend tiling at the same token count, which keeps the train and test
token geometry identical and the memory footprint inside 8 GB.

## Citation

A citation will be added once the associated article is published.

## License

MIT, see `LICENSE`.
