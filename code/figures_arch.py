"""Single system-architecture figure: the whole process flow of the study.

Layout, connectors and annotation are vector; the imagery is real, taken from a
held-out test scene and from the actual cached reconstructions, so the figure
shows what each stage genuinely produces rather than a stylised impression.
No equations appear in it.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from figures import save, COL2

EC = "#3A3A3A"
BLUE = "#DCE6F1"
GREEN = "#E4EFDC"
ORANGE = "#FBE5D6"
RED = "#F4DEDE"
GREY = "#EDEDED"
PURPLE = "#E6E0EF"
CACHE = os.path.join(C.RUNS, "cache")

DS = "ucmerced"
PREF_CLASSES = ["denseresidential", "harbor", "buildings", "mobilehomepark"]


# --------------------------------------------------------------- image data
def pick_scene():
    """A held-out test scene from a texture-rich class, where reconstruction
    differences are actually visible."""
    sp = C.get_splits(DS)
    classes = sp["classes"]
    for name in PREF_CLASSES:
        if name not in classes:
            continue
        ci = classes.index(name)
        for gi in sp["test"]:
            if sp["labels"][gi] == ci:
                return gi, name
    return sp["test"][0], classes[sp["labels"][sp["test"][0]]]


GI, GI_NAME = pick_scene()
_CROP = None


def thumb(domain, crop=True):
    """Return the chosen scene from a cached domain, centre-cropped so that the
    texture is legible at figure scale."""
    p = os.path.join(CACHE, DS + "_" + domain + ".npy")
    if not os.path.exists(p):
        return None
    a = np.load(p, mmap_mode="r")[GI]
    a = np.asarray(a)
    if crop:
        n = a.shape[0]
        k = n // 2
        o = (n - k) // 2
        a = a[o:o + k, o:o + k]
    return a


def place(ax, img, cx, cy, s, z=4, ec=EC, lw=0.7):
    """Draw a raster thumbnail inside the vector canvas."""
    if img is None:
        ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fc="#DDDDDD",
                               ec=ec, lw=lw, zorder=z))
        return
    ax.imshow(img, extent=[cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2],
              interpolation="nearest", zorder=z, aspect="auto")
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fc="none", ec=ec,
                           lw=lw, zorder=z + 2))


# ------------------------------------------------------------------ helpers
def box(ax, x0, x1, y0, y1, fc, lw=0.8, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.35,rounding_size=0.8",
                                fc=fc, ec=EC, lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, fs=6.8, w="normal", st="normal", c="#1A1A1A", ha="center", z=8):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, fontweight=w,
            style=st, color=c, zorder=z)


def arrow(ax, p0, p1, ls="-", col=EC, lw=0.8, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                                 lw=lw, ls=ls, color=col, shrinkA=0, shrinkB=0,
                                 zorder=z))


def elbow(ax, x0, y0, x1, y1, col=EC, ls="-", lw=0.55, z=3):
    xm = (x0 + x1) / 2.0
    ax.plot([x0, xm], [y0, y0], ls=ls, lw=lw, color=col, zorder=z,
            solid_capstyle="round")
    ax.plot([xm, xm], [y0, y1], ls=ls, lw=lw, color=col, zorder=z,
            solid_capstyle="round")
    arrow(ax, (xm, y1), (x1, y1), ls=ls, col=col, lw=lw, z=z)


def net_icon(ax, cx, cy, s):
    cols = [(-0.34, 4), (0.0, 5), (0.34, 3)]
    pts = []
    for dx, k in cols:
        ys = np.linspace(-0.34, 0.34, k)
        pts.append([(cx + dx * s, cy + y * s) for y in ys])
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], lw=0.25, color="#9AA6B5",
                        zorder=4)
    for layer, col in zip(pts, ["#7FA6D0", "#5D89BC", "#3F6EA5"]):
        for p in layer:
            ax.add_patch(Circle(p, 0.045 * s, fc=col, ec="none", zorder=5))


def bars_icon(ax, cx, cy, s):
    hs = [0.30, 0.55, 0.42, 0.72]
    cols = ["#4C72B0", "#4C72B0", "#C44E52", "#C44E52"]
    w = s * 0.16
    for i, (hh, cc) in enumerate(zip(hs, cols)):
        ax.add_patch(Rectangle((cx - s * 0.36 + i * s * 0.21, cy - s * 0.34),
                               w, hh * s, fc=cc, ec="none", zorder=5))
    ax.plot([cx - s * 0.42, cx + s * 0.42], [cy - s * 0.34, cy - s * 0.34],
            lw=0.6, color=EC, zorder=6)


# ------------------------------------------------------------------- figure
def fig_architecture():
    fig, ax = plt.subplots(figsize=(COL2, 4.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis("off")
    ax.set_facecolor("white")

    for x0, x1, name in [(1.5, 22.5, "1. Data and degradation"),
                         (24.5, 49.5, "2. Input domains, all size matched"),
                         (51.5, 74.5, "3. Two evaluation regimes"),
                         (76.5, 98.5, "4. Controlled measurement")]:
        ax.add_patch(Rectangle((x0, 62.6), x1 - x0, 4.0, fc="#F4F5F7",
                               ec="#C8CCD2", lw=0.5, zorder=1))
        txt(ax, (x0 + x1) / 2, 64.6, name, fs=7.2, w="bold", c="#33383F")

    # ------------------------------------------------------ stage 1
    place(ax, thumb("hr"), 12, 54.5, 11)
    txt(ax, 12, 47.2, "HR reference scene", fs=6.5, w="bold")

    box(ax, 3.5, 20.5, 34.0, 43.0, GREY)
    txt(ax, 12, 40.6, "Degradation", fs=6.9, w="bold")
    txt(ax, 12, 38.0, "P1  bicubic decimation", fs=6.1)
    txt(ax, 12, 36.0, "P2  blur, noise, JPEG", fs=6.1)
    arrow(ax, (12, 48.7), (12, 43.6))

    place(ax, thumb("lr_p1"), 12, 24.0, 7.0)
    txt(ax, 12, 18.8, "LR observation", fs=6.5, w="bold")
    arrow(ax, (12, 33.4), (12, 27.8))

    txt(ax, 12, 13.4, "degradation is applied after the", fs=5.6, st="italic",
        c="#5A5A5A")
    txt(ax, 12, 11.4, "split, so no test scene reaches", fs=5.6, st="italic",
        c="#5A5A5A")
    txt(ax, 12, 9.4, "any reconstruction network", fs=5.6, st="italic",
        c="#5A5A5A")

    # ------------------------------------------------------ stage 2
    rows = [(57.5, "hr", "HR reference", "oracle upper bound", GREEN),
            (50.0, "bic_p1", "Bicubic upsample", "the control: adds no information", BLUE),
            (42.5, "srcnn_p1", "SRCNN", "distortion oriented", BLUE),
            (35.0, "edsr_p1", "EDSR", "distortion oriented", BLUE),
            (27.5, "hfgan_p_p1", "HFGAN-P", "perception oriented", PURPLE),
            (20.0, "hfgan_g_p1", "HFGAN-G", "perception oriented, adversarial", PURPLE)]
    for y, dom, name, sub, fc in rows:
        box(ax, 26.0, 48.5, y - 3.2, y + 3.2, fc)
        place(ax, thumb(dom), 29.4, y, 5.0, lw=0.5)
        txt(ax, 39.8, y + 1.2, name, fs=6.6, w="bold")
        txt(ax, 39.8, y - 1.4, sub, fs=5.5, c="#4A4A4A")

    for y in [57.5]:
        elbow(ax, 17.8, 54.5, 25.5, y)
    for y in [50.0, 42.5, 35.0, 27.5, 20.0]:
        elbow(ax, 17.8, 24.0, 25.5, y)

    box(ax, 26.0, 48.5, 9.0, 15.0, "#FFFFFF", ls=(0, (2.2, 1.6)))
    place(ax, thumb("lr_p1"), 29.4, 12.0, 2.6, lw=0.5)
    txt(ax, 39.8, 13.2, "LR native", fs=6.4, w="bold")
    txt(ax, 39.8, 10.8, "not size matched: the", fs=5.5, c="#4A4A4A")
    txt(ax, 39.8, 9.4, "uncontrolled baseline", fs=5.5, c="#4A4A4A")
    arrow(ax, (17.8, 20.4), (25.5, 12.8), ls=(0, (2.2, 1.6)))

    # ------------------------------------------------------ stage 3
    box(ax, 53.0, 73.5, 41.5, 58.5, GREEN)
    txt(ax, 63.2, 56.4, "Matched regime", fs=7.0, w="bold")
    net_icon(ax, 58.4, 48.6, 7.6)
    txt(ax, 62.6, 51.6, "a classifier is trained", fs=5.7, ha="left")
    txt(ax, 62.6, 49.6, "on each domain and", fs=5.7, ha="left")
    txt(ax, 62.6, 47.6, "tested on that domain", fs=5.7, ha="left")
    txt(ax, 63.2, 43.4, "rebuilding the pipeline", fs=5.5, st="italic", c="#5A5A5A")

    box(ax, 53.0, 73.5, 20.5, 37.5, ORANGE)
    txt(ax, 63.2, 35.4, "Transfer regime", fs=7.0, w="bold")
    net_icon(ax, 58.4, 27.6, 7.6)
    txt(ax, 62.6, 30.6, "one HR-trained model", fs=5.7, ha="left")
    txt(ax, 62.6, 28.6, "is reused unchanged", fs=5.7, ha="left")
    txt(ax, 62.6, 26.6, "on every domain", fs=5.7, ha="left")
    txt(ax, 63.2, 22.4, "reusing an existing model", fs=5.5, st="italic", c="#5A5A5A")

    for y in [57.5, 50.0, 42.5, 35.0, 27.5, 20.0]:
        elbow(ax, 48.9, y, 52.6, 50.0)
        elbow(ax, 48.9, y, 52.6, 29.0)

    # ------------------------------------------------------ stage 4
    box(ax, 78.0, 97.5, 38.5, 58.5, RED)
    bars_icon(ax, 87.7, 54.2, 10)
    txt(ax, 87.7, 47.8, "Controlled comparison", fs=6.9, w="bold")
    txt(ax, 87.7, 45.2, "gain over the bicubic control", fs=5.8)
    txt(ax, 87.7, 43.2, "gap remaining to the HR reference", fs=5.8)
    txt(ax, 87.7, 41.2, "paired tests, three seeds, two backbones", fs=5.8)

    box(ax, 78.0, 97.5, 20.5, 35.0, "#EAF1F6")
    txt(ax, 87.7, 32.4, "Which metric predicts", fs=6.9, w="bold")
    txt(ax, 87.7, 29.6, "pixel fidelity against perceptual", fs=5.8)
    txt(ax, 87.7, 27.6, "distance, correlated per class", fs=5.8)
    txt(ax, 87.7, 25.0, "the reversal between the regimes", fs=5.5, st="italic",
        c="#5A5A5A")
    txt(ax, 87.7, 23.0, "identifies the mechanism", fs=5.5, st="italic",
        c="#5A5A5A")

    arrow(ax, (73.9, 50.0), (77.6, 48.6))
    arrow(ax, (73.9, 29.0), (77.6, 30.6))
    arrow(ax, (87.7, 38.1), (87.7, 35.6))

    # ------------------------------------------------------ outcome strip
    box(ax, 51.5, 97.5, 7.5, 16.5, "#FFF8E7", lw=0.9)
    txt(ax, 53.8, 14.2, "Outcome", fs=6.9, w="bold", ha="left")
    txt(ax, 53.8, 11.6, "retraining possible:  the reconstruction adds almost nothing over the control",
        fs=5.9, ha="left")
    txt(ax, 53.8, 9.5, "retraining impossible:  it adds a great deal, and perceptual quality predicts it",
        fs=5.9, ha="left")
    arrow(ax, (87.7, 20.1), (87.7, 17.1))

    save(fig, "fig_architecture")
    print("  scene used:", GI_NAME, "(test index", GI, ")")


if __name__ == "__main__":
    fig_architecture()
