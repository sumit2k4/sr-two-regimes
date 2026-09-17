"""Evaluation metrics: PSNR, SSIM, LPIPS.

Inputs to all functions are tensors in [-1, 1] of shape (B, 3, H, W).
PSNR/SSIM are computed on the [0, 1] range; LPIPS uses the standard AlexNet backend.
"""
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

try:
    import lpips as _lpips
    _LPIPS_NET = None
except Exception:
    _lpips = None


def _to_numpy01(t):
    x = ((t.detach().cpu().clamp(-1, 1) + 1) / 2).numpy()
    return np.transpose(x, (0, 2, 3, 1))  # B,H,W,C


def psnr(sr, hr):
    s, h = _to_numpy01(sr), _to_numpy01(hr)
    return float(np.mean([sk_psnr(h[i], s[i], data_range=1.0) for i in range(s.shape[0])]))


def ssim(sr, hr):
    s, h = _to_numpy01(sr), _to_numpy01(hr)
    vals = [sk_ssim(h[i], s[i], data_range=1.0, channel_axis=-1) for i in range(s.shape[0])]
    return float(np.mean(vals))


def lpips(sr, hr, device="cpu"):
    global _LPIPS_NET
    if _lpips is None:
        raise ImportError("Install the 'lpips' package to compute LPIPS.")
    if _LPIPS_NET is None:
        _LPIPS_NET = _lpips.LPIPS(net="alex").to(device).eval()
    with torch.no_grad():
        return float(_LPIPS_NET(sr.to(device), hr.to(device)).mean().item())
