"""Super-resolution model zoo spanning the perception-distortion trade-off.

  bicubic   : no learning (control)
  srcnn     : Dong et al. shallow CNN, operates on the bicubic-upsampled input
  edsr      : EDSR-baseline (16 residual blocks, 64 ch), distortion oriented
  hfgan_p   : the Paper-2 hybrid RRDB/Transformer generator trained with L1 +
              VGG perceptual loss only (no adversary)  -> distortion oriented
  hfgan_g   : the same generator with the full adversarial objective
              (this is the published Paper-2 model)    -> perception oriented

hfgan_* are imported verbatim from the Paper-2 code release so that the SR stage
of this study is exactly the previously published architecture.
"""
import sys, os, math
import torch, torch.nn as nn, torch.nn.functional as F

_P2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper2_ref",
                   "satellite-sr-code-release")
if _P2 not in sys.path:
    sys.path.insert(0, _P2)
from src.models import Generator as _P2Generator, Discriminator as _P2Discriminator  # noqa: E402


# --------------------------------------------------------------------- SRCNN
class SRCNN(nn.Module):
    needs_upsampled_input = True

    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4), nn.ReLU(True),
            nn.Conv2d(64, 32, 5, 1, 2), nn.ReLU(True),
            nn.Conv2d(32, 3, 5, 1, 2))

    def forward(self, lr):
        up = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return torch.clamp(self.body(up), 0.0, 1.0)


# ---------------------------------------------------------------------- EDSR
class _ResBlock(nn.Module):
    def __init__(self, nf=64, res_scale=1.0):
        super().__init__()
        self.c1 = nn.Conv2d(nf, nf, 3, 1, 1); self.c2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x), True)) * self.res_scale


class EDSR(nn.Module):
    needs_upsampled_input = False

    def __init__(self, scale=4, nf=64, nb=16):
        super().__init__()
        self.head = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[_ResBlock(nf) for _ in range(nb)],
                                  nn.Conv2d(nf, nf, 3, 1, 1))
        ups = []
        for _ in range(int(math.log2(scale))):
            ups += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2)]
        self.up = nn.Sequential(*ups)
        self.tail = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, lr):
        f = self.head(lr)
        f = f + self.body(f)
        return torch.clamp(self.tail(self.up(f)), 0.0, 1.0)


# ------------------------------------------------- Paper-2 hybrid fusion GAN
class HFGAN(nn.Module):
    """Wraps the Paper-2 Generator: [0,1] in/out, resolution-agnostic through
    bilinear interpolation of the learned positional embedding."""
    needs_upsampled_input = False

    def __init__(self, scale=4, img_size=32, num_blocks=16):
        super().__init__()
        assert scale == 4
        self.train_img_size = img_size
        self.g = _P2Generator(img_size=img_size, num_blocks=num_blocks)

    def _resize_pos(self, h, w):
        """Return positional embedding resampled to an h*w token grid."""
        tf = self.g.transformer
        s = tf.img_size
        pe = tf.pos_embedding                       # 1, s*s, D
        if h == s and w == s:
            return pe
        D = pe.shape[-1]
        pe = pe.reshape(1, s, s, D).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h, w), mode="bicubic", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, h * w, D)

    def _tf_forward(self, x):
        tf = self.g.transformer
        B, C, H, W = x.shape
        flat = x.flatten(2).transpose(1, 2)
        seq = tf.in_proj(flat) + self._resize_pos(H, W)
        out = tf.transformer_encoder(seq)
        return tf.out_proj(out).transpose(1, 2).view(B, C, H, W)

    def forward(self, lr):
        g = self.g
        x = lr * 2.0 - 1.0                          # generator is trained in [-1,1]
        f0 = g.lrelu(g.shallow_conv(x))
        f_rrdb = g.rrdb_stack(f0)
        f_res = f_rrdb + f0
        f_fuse = self._tf_forward(f_res) + f_rrdb if g.use_transformer else f_res
        f_ref = g.sarb(f_fuse) if g.use_sarb else f_fuse
        f_hff = g.hff_conv(torch.cat([f_rrdb, f_ref], 1)) if g.hff_conv is not None else f_ref
        o = g.lrelu(g.pixel_shuffle1(g.up_conv1(f_hff)))
        o = g.lrelu(g.pixel_shuffle2(g.up_conv2(o)))
        o = g.tanh(g.final_conv(o))
        return torch.clamp((o + 1.0) / 2.0, 0.0, 1.0)


class Discriminator(nn.Module):
    """Paper-2 discriminator made resolution-agnostic (global pooling head), so
    the same realism head can score both datasets and feed the QGE descriptor."""
    def __init__(self):
        super().__init__()
        d = _P2Discriminator(hr_size=256)
        self.features = d.features
        self.head = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(0.2, True),
                                  nn.Linear(1024, 1))

    def forward(self, x):
        f = self.features(x * 2.0 - 1.0)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.head(f)


def build_sr(name, scale=4, img_size=32):
    name = name.lower()
    if name == "srcnn":   return SRCNN(scale)
    if name == "edsr":    return EDSR(scale)
    if name in ("hfgan_p", "hfgan_g"): return HFGAN(scale, img_size=img_size)
    raise ValueError(name)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def tiled_sr(model, lr, tile=32, overlap=8, scale=4):
    """Overlap-and-blend tiled inference.

    The Paper-2 generator embeds absolute positional codes for a fixed token
    grid, so full-image inference on a 64x64 LR frame needs a 4096-token
    attention map (>7 GB). Tiling at the training token count keeps inference
    inside 8 GB and removes any train/test token-count mismatch; the Hann-window
    blend makes the result seam-free.
    """
    B, Cc, H, W = lr.shape
    if H <= tile and W <= tile:
        return model(lr)
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] != H - tile: ys.append(H - tile)
    if xs[-1] != W - tile: xs.append(W - tile)
    out = torch.zeros(B, 3, H * scale, W * scale, device=lr.device, dtype=torch.float32)
    acc = torch.zeros(1, 1, H * scale, W * scale, device=lr.device, dtype=torch.float32)
    t4 = tile * scale
    win = torch.hann_window(t4, periodic=False, device=lr.device).clamp_min(1e-3)
    w2 = (win[:, None] * win[None, :])[None, None]
    for y in ys:
        for x in xs:
            p = model(lr[:, :, y:y + tile, x:x + tile]).float()
            out[:, :, y * scale:y * scale + t4, x * scale:x * scale + t4] += p * w2
            acc[:, :, y * scale:y * scale + t4, x * scale:x * scale + t4] += w2
    return (out / acc.clamp_min(1e-6)).clamp(0, 1)
