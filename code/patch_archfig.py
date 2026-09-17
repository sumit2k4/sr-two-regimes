"""Insert the system-architecture figure as the paper's overview figure and
wire its generator into the build."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------ build script
p = os.path.join(HERE, "build_paper.sh")
s = io.open(p, encoding="utf-8").read()
if "figures_arch" not in s:
    s = s.replace("$PY -u figures_transfer.py",
                  "$PY -u figures_transfer.py\n$PY -u figures_arch.py")
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("build_paper.sh: figures_arch wired in")

# ------------------------------------------------------------- manuscript
p = os.path.join(HERE, "..", "paper", "main.tex")
s = io.open(p, encoding="utf-8").read()

FIG = r"""
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{fig_architecture.pdf}
\caption{Process flow of the study. A held-out high-resolution scene is degraded
into a low-resolution observation, which is then expanded into six inputs that
are all presented at the same spatial size: the high-resolution reference, the
bicubic control, and four reconstructions spanning the perception-distortion
trade-off. The low-resolution frame itself is retained separately as the
uncontrolled baseline, and is drawn smaller here because that size difference is
exactly the confound the control removes. Every domain is then evaluated in both
regimes, and the two regimes are compared against the same bicubic control. All
imagery is real: the thumbnails are the same test scene taken from the
corresponding cached domain.}
\label{fig:arch}
\end{figure*}
"""

anchor = "\\section{Two Regimes and a Controlled Protocol}\n\\label{sec:prob}\n"
assert anchor in s
s = s.replace(anchor, anchor + FIG.strip("\n") + "\n\n", 1)

# reference it from the introduction
old = """The contributions of this article are as follows."""
new = """Fig.~\\ref{fig:arch} summarises the resulting process flow, and the
contributions of this article are as follows."""
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("architecture figure inserted as Fig. 1")
