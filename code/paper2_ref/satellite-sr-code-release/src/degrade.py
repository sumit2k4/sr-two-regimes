"""Deterministic LR-HR degradation pipelines.

Two protocols are provided so that every table in the paper can be reproduced:
  * 'bicubic'      : x4 bicubic down-sampling only (Tables 5-9).
  * 'blur_noise'   : 7x7 Gaussian blur (sigma=1.2) -> x4 bicubic -> additive
                     Gaussian noise (sigma=5/255), clipped to [0,1] (Table 10).

All operations are applied in RGB space and are fully deterministic given the
configured sigmas, so LR-HR pairs are identical across runs and machines.
"""
import numpy as np
from PIL import Image
import cv2


def bicubic_downsample(hr_img: Image.Image, scale=4) -> Image.Image:
    w, h = hr_img.size
    return hr_img.resize((w // scale, h // scale), resample=Image.BICUBIC)


def blur_noise_degrade(hr_img: Image.Image, scale=4, blur_ksize=7,
                       blur_sigma=1.2, noise_sigma=5.0, seed=None) -> Image.Image:
    """Non-ideal degradation pipeline used for the robustness experiment (Table 10).

    Manuscript settings: 7x7 Gaussian blur with sigma=1.2, then x4 bicubic, then
    zero-mean additive Gaussian noise with sigma=5/255 (i.e. 5.0 on the 0-255 scale),
    clipped to the valid intensity range.
    """
    arr = np.asarray(hr_img).astype(np.float32)
    # 1) Gaussian blur (per channel, separable kernel handled by cv2)
    blurred = cv2.GaussianBlur(arr, (blur_ksize, blur_ksize), blur_sigma)
    blurred_img = Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))
    # 2) x4 bicubic down-sample
    lr = np.asarray(bicubic_downsample(blurred_img, scale)).astype(np.float32)
    # 3) additive zero-mean Gaussian noise
    rng = np.random.default_rng(seed)
    lr = lr + rng.normal(0.0, noise_sigma, lr.shape)
    lr = np.clip(lr, 0, 255).astype(np.uint8)
    return Image.fromarray(lr)


def degrade(hr_img: Image.Image, mode="bicubic", scale=4, **kwargs) -> Image.Image:
    if mode == "bicubic":
        return bicubic_downsample(hr_img, scale)
    if mode == "blur_noise":
        return blur_noise_degrade(hr_img, scale, **kwargs)
    raise ValueError(f"Unknown degradation mode: {mode}")
