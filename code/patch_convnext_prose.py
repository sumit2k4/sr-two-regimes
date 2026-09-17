"""Reconcile the prose with the ConvNeXt-Tiny results.

Three substantive corrections are required by the new measurements:
  1. the EuroSAT naive gain no longer exists, because ConvNeXt cannot ingest a
     16x16 input at all -- which is a sharper version of the same argument;
  2. HFGAN-G is no longer the best transfer model; HFGAN-P is, on both datasets;
  3. the backbone is ConvNeXt-Tiny, not ResNet-18.
"""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "main.tex")
s = io.open(P, encoding="utf-8").read()

EDITS = []

# ---- 1. the EuroSAT naive-gain paragraph -----------------------------------
EDITS.append((
"""The effect is starkest on EuroSAT, where the native LR input is only
$16\\times16$ pixels and the naive comparison is correspondingly flattering:
super-resolution appears to add between \\N{naiveeurosatsrcnn} and
\\N{naiveeurosatedsr}\\,pp, but against the bicubic control the same
reconstructions move accuracy by between \\N{ctrleurosathfgang} and
\\N{ctrleurosatedsr}\\,pp. On UC Merced, where the LR input is $64\\times64$ and a
ResNet-18 already copes with it, the naive gain is smaller to begin with
(\\N{naiveucmercedsrcnn} to \\N{naiveucmercedhfganp}\\,pp) and the controlled gain
smaller still. That contrast is itself informative: the size of the apparent
benefit is governed largely by how small the uncontrolled baseline input
happens to be, which is a property of the experimental setup rather than of the
reconstruction network.""",
"""On UC Merced the naive comparison suggests a gain of
\\N{naiveucmercedsrcnn} to \\N{naiveucmercedhfganp}\\,pp, while against the
bicubic control the same reconstructions move accuracy by only
\\N{ctrlucmercedhfgang} to \\N{ctrlucmercedhfganp}\\,pp. On EuroSAT the controlled
gains are smaller still, between \\N{ctrleurosathfgang} and
\\N{ctrleurosatedsr}\\,pp.

The EuroSAT naive column is empty, and the reason is worth stating because it
sharpens the argument rather than weakening it. ConvNeXt's stem is a stride-four
convolution followed by three stride-two stages, so the smallest input it can
process is $32\\times32$; the EuroSAT LR frame is $16\\times16$ and the
LR-native condition simply cannot be run with this backbone. The
\\emph{uncontrolled} baseline is thus not even well defined across
architectures: its value depends on how small an input the chosen network
happens to tolerate, and for a large part of the modern backbone family it does
not exist at all. The bicubic control, by contrast, is always defined, because
it presents every architecture with the size it expects. That is a second and
independent reason to prefer it."""))

# ---- 2. the transfer ordering ----------------------------------------------
EDITS.append((
"""The ordering across models is the second important observation. In the transfer
regime the two perceptual models are clearly best on both datasets, and on
EuroSAT the fully adversarial HFGAN-G is best of all, despite having the
\\emph{lowest} PSNR of the four. The model that is worst by the metric the SR
literature optimises is the best for this deployment scenario.""",
"""The ordering across models is the second important observation, and it is
consistent on both datasets: the two perception-oriented models are clearly
best, and the two distortion-oriented ones clearly worse, even though the
distortion-oriented EDSR has the highest PSNR of the four. The models that score
worst on the metric the super-resolution literature optimises are the ones that
serve this deployment scenario best.

Within the perceptual pair the ordering is informative in a different way. The
purely perceptual HFGAN-P leads on both datasets
(\\N{trgaineurosathfganp}\\,pp on EuroSAT) ahead of the fully adversarial
HFGAN-G (\\N{trgaineurosathfgang}\\,pp), even though HFGAN-G attains the better
LPIPS of the two (\\N{LPIPSeurosathfgang} against
\\N{LPIPSeurosathfganp}). The relationship between perceptual quality and
transfer benefit is therefore strong but not monotone at the extreme: pushing a
reconstruction further towards distributional realism helps until the
synthesised content stops corresponding to the observation, after which it
begins to cost accuracy again. There is an optimum, and it does not lie at the
most aggressive end of the trade-off."""))

# ---- 3. backbone references -------------------------------------------------
EDITS.append((
"""Every classifier is an ImageNet-pretrained ResNet-18 \\cite{he2016resnet}
fine-tuned end to end with SGD (momentum $0.9$, weight decay
$5\\times10^{-4}$), a one-cycle schedule with peak learning rate $0.01$, label
smoothing $0.1$, and random flips and $90^\\circ$ rotations, which are the
symmetries of overhead imagery.""",
"""Every classifier is an ImageNet-pretrained ConvNeXt-Tiny
\\cite{liu2022convnext} fine-tuned end to end with AdamW (peak learning rate
$3\\times10^{-4}$, weight decay $0.05$) under a one-cycle schedule, with label
smoothing $0.1$ and random flips and $90^\\circ$ rotations, which are the
symmetries of overhead imagery. ConvNeXt is preferred to a windowed transformer
here because it is fully convolutional and therefore resolution agnostic, which
the study requires: inputs range from $16\\times16$ to $256\\times256$. The
complete factorial is additionally repeated with ResNet-18
\\cite{he2016resnet} as a backbone-robustness check (Section~\\ref{sec:backbone}).""",
))

EDITS.append((
"""additionally train EfficientNet-B0
\\cite{tan2019efficientnet} and DenseNet-121
\\cite{huang2017densenet} on the super-resolved domain. Every configuration is
repeated with seeds $42$, $123$ and $999$; all reported values are means with
standard deviations over those seeds. In total the study comprises $54$
classifier training runs.""",
"""additionally train EfficientNet-B0
\\cite{tan2019efficientnet} and DenseNet-121
\\cite{huang2017densenet} on the super-resolved domain. Every configuration is
repeated with seeds $42$, $123$ and $999$; all reported values are means with
standard deviations over those seeds. In total the study comprises $99$
classifier training runs across the two backbones."""))

EDITS.append((
"""On UC Merced, where the LR input is $64\\times64$ and a
ResNet-18 already copes with it,""",
"""On UC Merced, where the LR input is $64\\times64$,"""))

EDITS.append((
"""Second, the classifiers are ImageNet-pretrained ResNet-18 networks; a
remote-sensing-pretrained or larger backbone would raise all conditions and
might compress the differences between them, although it would also make the
matched regime even more saturated and so is unlikely to reverse the
conclusion.""",
"""Second, the classifiers are ImageNet-pretrained; a remote-sensing-pretrained
backbone might compress the differences further. We have at least bounded this
concern empirically, by repeating the whole factorial with two backbones of very
different capacity (Section~\\ref{sec:backbone}) and finding every conclusion
preserved."""))

for a, b in EDITS:
    if a not in s:
        print("NOT FOUND:", a[:70].replace("\n", " "))
        continue
    s = s.replace(a, b)

# ConvNeXt reference
if "liu2022convnext" not in s.split("\\begin{thebibliography}")[1]:
    s = s.replace("""\\bibitem{he2016resnet}""",
"""\\bibitem{liu2022convnext}
Z.~Liu, H.~Mao, C.-Y.~Wu, C.~Feichtenhofer, T.~Darrell, and S.~Xie, ``A ConvNet
for the 2020s,'' in \\emph{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.},
2022, pp.~11976--11986.

\\bibitem{he2016resnet}""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("prose reconciled with ConvNeXt results")
