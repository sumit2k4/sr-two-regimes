"""Bring the abstract inside the 200-220 word house rule without dropping any
substantive claim."""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "main.tex")

NEW = r"""Single-image super-resolution is routinely offered as a preprocessing step that
improves downstream remote sensing analysis, on evidence that is usually a
single comparison against the low-resolution input, which conflates
reconstructed detail with classifier input size and collapses two deployment
regimes that behave in opposite ways. We separate them with a controlled
factorial protocol: a low-resolution scene, its bicubic upsampling, four
reconstructions spanning the perception-distortion trade-off, and the
high-resolution reference reach identically trained classifiers at the same
spatial size, on UC Merced and EuroSAT, across two backbones and three seeds. When the classifier is retrained on the
reconstructions it will see, super-resolution contributes essentially nothing
beyond bicubic interpolation: controlled gains span \N{ctrlucmercedhfgang} to
\N{ctrlucmercedhfganp} percentage points, none of them significantly positive,
and the only significant effect is an adversarial model performing worse than
the control. When an existing high-resolution-trained classifier is reused
instead, the same reconstructions deliver up to \N{trgaineurosathfganp} points,
every gain significant. The two regimes also disagree about which reconstruction
metric matters: pixel fidelity predicts the first, perceptual distance the
second. Together these identify the mechanism as distributional alignment rather
than information recovery. Adaptive per-image fusion is also tested, and fails
for reasons an oracle bound explains. The study yields a reusable protocol and a
concrete decision rule."""


def count(a):
    plain = re.sub(r"\\[A-Za-z]+\*?", " ", a)
    plain = re.sub(r"[{}$\\~^_]", " ", plain)
    return len(plain.split())


s = io.open(P, encoding="utf-8").read()
head, rest = s.split("\\begin{abstract}", 1)
old, tail = rest.split("\\end{abstract}", 1)
print("old:", count(old), "words -> new:", count(NEW), "words")
assert 200 <= count(NEW) <= 220, "outside the 200-220 rule"
io.open(P, "w", encoding="utf-8", newline="\n").write(
    head + "\\begin{abstract}\n" + NEW.strip("\n") + "\n\\end{abstract}" + tail)
print("abstract replaced")
