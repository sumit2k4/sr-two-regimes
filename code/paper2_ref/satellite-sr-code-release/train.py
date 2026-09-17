"""Training entry point.

Example:
    python train.py --config configs/default.yaml --seed 42 \
        --train-split splits/ucmerced_train.txt --val-split splits/ucmerced_val.txt \
        --out runs/ucmerced_seed42

Three seeds (42, 123, 999) are used in the paper; results are reported as
mean +/- std over the three runs.
"""
import os
import argparse
import csv
import torch
import torch.optim as optim

from src.utils import set_seed, load_config
from src.models import Generator, Discriminator, count_parameters
from src.losses import HybridLossEvaluator
from src.dataset import build_loader
from src import metrics


def evaluate(netG, loader, device, use_lpips=False):
    netG.eval()
    ps, ss, lp = [], [], []
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = netG(lr)
            ps.append(metrics.psnr(sr, hr))
            ss.append(metrics.ssim(sr, hr))
            if use_lpips:
                lp.append(metrics.lpips(sr, hr, device))
    netG.train()
    out = {"psnr": sum(ps) / len(ps), "ssim": sum(ss) / len(ss)}
    if use_lpips and lp:
        out["lpips"] = sum(lp) / len(lp)
    return out


def variant_flags(variant):
    """Ablation variants for Table 9.
       I   = CNN only (no Transformer/SARB/HFF, no GAN), pure L1
       II  = CNN + Transformer fusion (SARB+HFF on), no GAN
       III = CNN + GAN (no Transformer/SARB/HFF)
       IV  = full proposed model (all components + GAN)
    """
    table = {
        "I":   dict(use_transformer=False, use_sarb=False, use_hff=False, use_gan=False),
        "II":  dict(use_transformer=True,  use_sarb=True,  use_hff=True,  use_gan=False),
        "III": dict(use_transformer=False, use_sarb=False, use_hff=False, use_gan=True),
        "IV":  dict(use_transformer=True,  use_sarb=True,  use_hff=True,  use_gan=True),
    }
    return table[variant]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-split", required=True)
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--variant", choices=["I", "II", "III", "IV"], default="IV",
                    help="Ablation variant (Table 9). Default IV = full proposed model.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flags = variant_flags(args.variant)
    use_gan = flags["use_gan"]

    netG = Generator(num_blocks=cfg["model"]["rrdb_blocks"],
                     embed_dim=cfg["model"]["embed_dim"],
                     num_heads=cfg["model"]["num_heads"],
                     num_layers=cfg["model"]["num_layers"],
                     use_transformer=flags["use_transformer"],
                     use_sarb=flags["use_sarb"],
                     use_hff=flags["use_hff"]).to(device)
    netD = Discriminator(hr_size=cfg["data"]["hr_size"]).to(device) if use_gan else None
    print(f"Variant {args.variant} | use_gan={use_gan} | "
          f"Generator params: {count_parameters(netG):,}"
          + (f" | Discriminator params: {count_parameters(netD):,}" if use_gan else ""))

    loss = HybridLossEvaluator(cfg["loss"]["lambda_pixel"], cfg["loss"]["lambda_percep"],
                               cfg["loss"]["lambda_adv"], device=device)
    optG = optim.Adam(netG.parameters(), lr=cfg["train"]["lr"], betas=(0.9, 0.999))
    schG = optim.lr_scheduler.StepLR(optG, step_size=cfg["train"]["lr_decay_every"], gamma=0.5)
    if use_gan:
        optD = optim.Adam(netD.parameters(), lr=cfg["train"]["lr"], betas=(0.9, 0.999))
        schD = optim.lr_scheduler.StepLR(optD, step_size=cfg["train"]["lr_decay_every"], gamma=0.5)

    train_loader = build_loader(args.train_split, cfg["train"]["batch_size"],
                                cfg["data"]["hr_size"], cfg["data"]["scale"],
                                cfg["data"]["degradation"], augment=True, shuffle=True)
    val_loader = build_loader(args.val_split, 1, cfg["data"]["hr_size"], cfg["data"]["scale"],
                              cfg["data"]["degradation"], augment=False, shuffle=False)

    log_path = os.path.join(args.out, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "g_loss", "d_loss", "val_psnr", "val_ssim"])

    best_psnr = -1.0
    for epoch in range(cfg["train"]["epochs"]):
        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = netG(lr)
            if use_gan:
                # --- Discriminator ---
                optD.zero_grad()
                d_loss = loss.discriminator_loss(netD(hr), netD(sr.detach()))
                d_loss.backward()
                optD.step()
                # --- Generator (pixel + perceptual + adversarial) ---
                optG.zero_grad()
                g_loss, _, _, _ = loss.generator_loss(sr, hr, netD(sr))
                g_loss.backward()
                optG.step()
            else:
                # --- Generator only (pixel + perceptual, no adversarial term) ---
                optG.zero_grad()
                l_pixel = loss.l1_loss(sr, hr)
                l_percep = loss.perceptual_loss(sr, hr)
                g_loss = loss.lambda_pixel * l_pixel + loss.lambda_percep * l_percep
                g_loss.backward()
                optG.step()
                d_loss = torch.tensor(0.0)
        schG.step()
        if use_gan:
            schD.step()

        val = evaluate(netG, val_loader, device)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{g_loss.item():.4f}", f"{d_loss.item():.4f}",
                                    f"{val['psnr']:.4f}", f"{val['ssim']:.4f}"])
        print(f"[{epoch+1}/{cfg['train']['epochs']}] G={g_loss.item():.4f} D={d_loss.item():.4f} "
              f"val_PSNR={val['psnr']:.3f} val_SSIM={val['ssim']:.4f}")

        # checkpoint selection = best validation PSNR
        if val["psnr"] > best_psnr:
            best_psnr = val["psnr"]
            torch.save(netG.state_dict(), os.path.join(args.out, "generator_best.pth"))
    torch.save(netG.state_dict(), os.path.join(args.out, "generator_last.pth"))
    print(f"Done. Best val PSNR = {best_psnr:.3f} (saved as generator_best.pth)")


if __name__ == "__main__":
    main()
