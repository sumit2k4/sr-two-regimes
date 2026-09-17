"""LR generation. Two deterministic protocols, both applied AFTER splitting.

  P1 'bicubic'    : x4 bicubic decimation (the idealised protocol used by almost
                    all remote-sensing SR papers, and by Paper 2 Tables 5-9).
  P2 'blur_noise' : 7x7 Gaussian blur (sigma=1.2) -> x4 bicubic -> AWGN
                    (sigma=5/255) -> JPEG q=85, i.e. optical blur, sensor noise
                    and on-board lossy coding, matching an operational EO chain.

Every SR model is trained on P1 only; P2 is a held-out degradation shift.
"""
import io
import numpy as np
from PIL import Image


def _center_crop_multiple(img: Image.Image, m: int) -> Image.Image:
    w, h = img.size
    nw, nh = (w // m) * m, (h // m) * m
    if (nw, nh) == (w, h):
        return img
    l, t = (w - nw) // 2, (h - nh) // 2
    return img.crop((l, t, l + nw, t + nh))


def make_hr(img: Image.Image, hr_size: int) -> Image.Image:
    """Canonical HR reference: centre-crop to a multiple of the scale then resize
    to the dataset HR size (no-op for images already at hr_size)."""
    img = _center_crop_multiple(img, 4)
    if img.size != (hr_size, hr_size):
        img = img.resize((hr_size, hr_size), Image.BICUBIC)
    return img


def _gauss_kernel(k, sigma):
    ax = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2)); g /= g.sum()
    return g


def _blur(arr, k=7, sigma=1.2):
    g = _gauss_kernel(k, sigma); p = k // 2
    a = np.pad(arr, ((p, p), (p, p), (0, 0)), mode="reflect")
    tmp = np.zeros_like(arr, dtype=np.float64)
    for i in range(k):
        tmp += g[i] * a[i:i + arr.shape[0], p:p + arr.shape[1], :]
    a2 = np.pad(tmp, ((p, p), (p, p), (0, 0)), mode="reflect")
    out = np.zeros_like(tmp)
    for j in range(k):
        out += g[j] * a2[p:p + arr.shape[0], j:j + arr.shape[1], :]
    return out


def degrade(hr: Image.Image, mode="bicubic", scale=4, seed=0) -> Image.Image:
    w, h = hr.size
    if mode == "bicubic":
        return hr.resize((w // scale, h // scale), Image.BICUBIC)
    if mode == "blur_noise":
        arr = np.asarray(hr, dtype=np.float64)
        blurred = np.clip(_blur(arr), 0, 255).astype(np.uint8)
        lr = Image.fromarray(blurred).resize((w // scale, h // scale), Image.BICUBIC)
        a = np.asarray(lr, dtype=np.float64)
        rng = np.random.RandomState(seed)
        a = np.clip(a + rng.normal(0.0, 5.0, a.shape), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(mode)


def bicubic_up(lr: Image.Image, scale=4) -> Image.Image:
    w, h = lr.size
    return lr.resize((w * scale, h * scale), Image.BICUBIC)
