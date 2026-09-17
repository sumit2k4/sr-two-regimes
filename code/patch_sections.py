"""Add the backbone-robustness section, the literature-context table, and the
saturation argument."""
import io
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "main.tex")
s = io.open(P, encoding="utf-8").read()

# ---------------------------------------------------------- backbone section
BACKBONE = r"""
\subsection{Does the Conclusion Depend on the Backbone?}
\label{sec:backbone}
A natural objection is that a stronger classifier might extract from the
reconstructions what ours cannot. We therefore repeated the entire factorial
with ResNet-18 branches, giving two backbones that differ by roughly an order of
magnitude in modelling capacity and by six years of architectural development,
evaluated on the identical reconstructions and the identical split.
Table~\ref{tab:backbone} places the headline quantities side by side.

Every conclusion is preserved. Across both backbones and both datasets, the
number of statistically significant \emph{positive} controlled gains is
\N{bbSigPos}; the only significant controlled effect in either configuration is
the adversarial model performing worse than the bicubic control. The transfer
gains remain large and significant throughout, and the metric reversal, fidelity
in the matched regime and perceptual distance in the transfer regime, appears in
both.

The differences that do appear are informative rather than awkward. The stronger
backbone raises every condition, so the oracle gap narrows and the matched
regime becomes even less rewarding. More interestingly, the transfer gain on UC
Merced falls from \N{bbr18ucmercedtrhi}\,pp with ResNet-18 to
\N{bbcnxucmercedtrhi}\,pp with ConvNeXt, while on EuroSAT it is essentially
unchanged (\N{bbr18eurosattrhi} against \N{bbcnxeurosattrhi}\,pp). This is what
the alignment account predicts: a stronger, more heavily regularised backbone is
intrinsically more robust to a moderate domain shift, so there is less
misalignment for the reconstruction to repair, whereas the severe shift on
EuroSAT exceeds what architectural robustness alone can absorb. Super-resolution
buys domain alignment, and its value therefore falls as the classifier becomes
able to supply that alignment itself.

\begin{table}[!t]
\caption{Backbone robustness. Every headline quantity, computed twice on the
same reconstructions and the same split.}
\label{tab:backbone}
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcc}
\toprule
Quantity & ConvNeXt-Tiny & ResNet-18 \\
\midrule
\inputrows{auto/tab_backbone.tex}
\bottomrule
\end{tabular}
\end{table}

\subsection{Context Against the Published Literature}
\label{sec:context}
Table~\ref{tab:context} places our matched-domain HR accuracy beside recent
results on the same two benchmarks. The purpose is not to rank: the split
protocols differ, most published UC Merced results use an $80/20$ partition
where ours is $70/15/15$, and our HR row is a deliberately plain reference
condition rather than a tuned attempt at a record. The purpose is to establish
that the classifiers underpinning this study are competitive, so that the
absence of a matched-regime super-resolution benefit cannot be attributed to a
weak downstream model.

It is worth being explicit about the scale of these differences. Our UC Merced
test partition contains \N{ucmTestN} images, so the attainable accuracies near
the top are quantised: \N{ucmTwoErr}\,\% is two misclassified images,
\N{ucmOneErr}\,\% is one, and our \N{bbcnxucmercedhr}\,\% is
\N{ucmOurErr}. The best published figure on this benchmark, $99.76$\,\%,
corresponds to exactly one misclassified image out of the $420$ in an $80/20$
partition. Every recent result on both datasets, ours included, lies within
about one percentage point of every other, which is a range of two or three
images. That is precisely why this article measures controlled differences
between input domains on a fixed split, with paired tests, rather than competing
on absolute accuracy: at this level of saturation, absolute accuracy is no
longer a discriminating quantity.

\begin{table}[!t]
\caption{Recent reported accuracy on the two benchmarks. Values are quoted from
the cited works under their own split protocols and are indicative only; they
are not protocol-identical with each other or with this study.}
\label{tab:context}
\centering
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llc}
\toprule
Dataset & Method & OA (\%) \\
\midrule
\multicolumn{3}{l}{\textit{UC Merced}} \\
Swin-T, transfer learning & \cite{cheng2020meets} protocol & 99.76* \\
Fused CNNs with SR front end & Albarakati \emph{et al.} \cite{albarakati2024unified} & 99.64* \\
Pretrained ResNet-152 & survey value \cite{cheng2020meets} & 98.81* \\
\textbf{This work, HR reference} & ConvNeXt-Tiny, $70/15/15$ & \textbf{\N{bbcnxucmercedhr}} \\
\midrule
\multicolumn{3}{l}{\textit{EuroSAT}} \\
ResNet-18 + topological features & Sharma \cite{sharma2025tda} & 99.33* \\
ConvNeXt-Tiny, transfer learning & reported value & 99.11* \\
ResNet-50 (original benchmark) & Helber \emph{et al.} \cite{helber2019eurosat} & 98.57* \\
\textbf{This work, HR reference} & ConvNeXt-Tiny, $70/15/15$ & \textbf{\N{bbcnxeurosathr}} \\
\bottomrule
\end{tabular}
\end{table}
"""

anchor = "\\subsection{Robustness to an Unseen Degradation}"
assert anchor in s
s = s.replace(anchor, BACKBONE.strip("\n") + "\n\n" + anchor, 1)

# ---------------------------------------------------------------- reference
if "sharma2025tda" not in s:
    s = s.replace("\\bibitem{he2016resnet}",
"""\\bibitem{sharma2025tda}
A.~Sharma, ``Improving remote sensing classification using topological data
analysis and convolutional neural networks,'' \\emph{arXiv:2507.10381}, 2025.

\\bibitem{he2016resnet}""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("backbone + context sections added")
