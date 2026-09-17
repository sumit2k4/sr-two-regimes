"""Bring the abstract and two captions into line with the measured correlation
result, which came out opposite to the drafted hypothesis."""
import io
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "main.tex")
s = io.open(P, encoding="utf-8").read()

EDITS = [
    ("We further show that the change in pixel fidelity does\n"
     "not predict the change in classification accuracy, whereas the change in\n"
     "perceptual distance does.",
     "The only statistically significant controlled effect we observe is an\n"
     "adversarial model performing worse than bicubic interpolation, and the learned\n"
     "perceptual metric on which such models are optimised does not predict the change\n"
     "in classification accuracy, while pixel-fidelity metrics retain some predictive\n"
     "value."),
    ("Negative $\\rho$ for $\\Delta$LPIPS indicates that a \\emph{decrease} in perceptual\n"
     "distance accompanies an \\emph{increase} in accuracy.",
     "A positive $\\rho$ means that an increase in the quantity accompanies an increase\n"
     "in accuracy. For $\\Delta$LPIPS, where lower is perceptually better, a positive\n"
     "$\\rho$ would indicate that perceptual degradation accompanies improvement."),
    ("Pixel\n"
     "fidelity carries little information about the downstream outcome; perceptual\n"
     "distance carries much more.",
     "On EuroSAT the fidelity metrics carry information about the\n"
     "downstream outcome while the perceptual metric does not; on the saturated\n"
     "UC Merced benchmark no metric does."),
]

for a, b in EDITS:
    assert a in s, a[:60]
    s = s.replace(a, b)

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("abstract and captions updated")
