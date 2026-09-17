"""House style: captions are short titles (under ten words). Everything the
caption used to explain moves to a note directly beneath the float, so no
information is lost.

Run with --list to inventory captions, or with no argument to apply.
"""
import io
import os
import re
import sys

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "main.tex")
CAP = "\\caption{"


def spans(s):
    """Yield (start, inner_start, inner_end, end, label) for every caption."""
    out = []
    i = 0
    while True:
        i = s.find(CAP, i)
        if i < 0:
            break
        j = i + len(CAP)
        depth = 1
        k = j
        while depth and k < len(s):
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
            k += 1
        m = re.search(r"\\label\{([^}]+)\}", s[k:k + 400])
        out.append((i, j, k - 1, k, m.group(1) if m else "?"))
        i = k
    return out


# label -> (short caption <=10 words, note placed beneath the float)
NEW = {
 "fig:arch": ("Process flow of the controlled two-regime study.",
  "A held-out high-resolution scene is degraded into a low-resolution "
  "observation, then expanded into six inputs presented at the same spatial "
  "size: the high-resolution reference, the bicubic control, and four "
  "reconstructions spanning the perception-distortion trade-off. The "
  "low-resolution frame is retained separately as the uncontrolled baseline and "
  "drawn smaller, because that size difference is exactly the confound the "
  "control removes. Every domain enters both regimes. All thumbnails are the "
  "same UC~Merced test scene taken from the corresponding cached domain."),
 "tab:srq": ("Reconstruction quality, test split, protocol P1.",
  "Scale factor four. Mean and standard deviation are taken over test images."),
 "tab:confound": ("Matched regime: gains against two baselines.",
  "Each classifier is trained and tested on the same domain. OA is overall "
  "accuracy in per cent, mean and standard deviation over three seeds. "
  "$\\Delta_{\\mathrm{naive}}$ is measured against the native low-resolution "
  "input, $\\Delta_{\\mathrm{ctrl}}$ against the bicubic control, and "
  "$\\Delta_{\\mathrm{oracle}}$ is the gap remaining to the high-resolution "
  "reference. The last column is the McNemar $p$ value against the control."),
 "fig:tworegimes": ("The same reconstructions, measured two ways.",
  "Blue: the matched regime, in which the classifier is retrained on the "
  "reconstruction it will see. Red: the transfer regime, in which a "
  "high-resolution-trained classifier is reused unchanged. Both are measured "
  "against the bicubic control, so both isolate reconstructed content from "
  "input size."),
 "tab:transfer": ("Transfer regime: one classifier, every input.",
  "A single classifier trained on genuine high-resolution imagery is applied "
  "unchanged to each reconstruction. Gains are relative to the bicubic control; "
  "$p$ is McNemar against that control."),
 "tab:corr": ("Matched regime: metric change against accuracy change.",
  "Correlation between the per-class change in reconstruction quality and the "
  "per-class change in accuracy, pooled over the four SR models. A positive "
  "$\\rho$ means an increase in the quantity accompanies an increase in "
  "accuracy; for $\\Delta$LPIPS, where lower is perceptually better, a positive "
  "$\\rho$ would indicate that perceptual degradation accompanies improvement."),
 "tab:transfercorr": ("Transfer regime: metric change against accuracy change.",
  "As the preceding table, for the high-resolution-trained classifier. A "
  "negative $\\rho$ for $\\Delta$LPIPS means that a reduction in perceptual "
  "distance accompanies an increase in accuracy."),
 "fig:lpipstransfer": ("Perceptual distance against accuracy, both regimes.",
  "Per-class change in perceptual distance against per-class change in "
  "accuracy. LPIPS carries no information about the matched-regime outcome and "
  "predicts the transfer outcome."),
 "fig:perclass": ("Per-class gain of HFGAN-G over the control.",
  "Matched regime, UC~Merced, sorted by the change."),
 "tab:perclass": ("Per-class accuracy on UC Merced, matched regime.",
  "Accuracy in per cent under the bicubic control, under HFGAN-G, their "
  "difference, and the high-resolution reference."),
 "fig:matrix": ("Accuracy for every training and test domain.",
  "UC~Merced. The diagonal is the matched regime and the top row is the "
  "transfer regime. P2 columns are the unseen operational degradation."),
 "tab:backbone": ("Backbone robustness of every headline quantity.",
  "Each quantity is computed twice, on the same reconstructions and the same "
  "split, once with ConvNeXt-Tiny branches and once with ResNet-18."),
 "tab:context": ("Recent reported accuracy on the two benchmarks.",
  "Values are quoted from the cited works under their own split protocols and "
  "are indicative only; they are not protocol-identical with each other or with "
  "this study. The train/test partition is named where the source states it and "
  "marked \\emph{split not stated} where it does not, which is itself part of "
  "the difficulty of comparing across this literature. "
  "\\textsuperscript{\\dag}The $80/20$ partition for the Swin-T entry is "
  "inferred rather than quoted: $99.76$\\,\\% is $1-1/420$, and $420$ is the "
  "test size of an $80/20$ partition of UC~Merced."),
 "fig:framework": ("The quality-gated ensemble under test.",
  "A gate reads a ground-truth-free reliability descriptor and emits per-image "
  "mixing weights over three reconstructions of the same scene."),
 "tab:qgep1": ("Fusion strategies, protocol P1, matched regime.",
  "Mean and standard deviation over three seeds; ECE is expected calibration "
  "error."),
 "tab:qgep2": ("Fusion strategies under the unseen degradation.",
  "Protocol P2 applies blur, decimation, sensor noise and JPEG coding. No "
  "super-resolution model was trained on this chain."),
 "tab:headroom": ("Headroom available to any per-image gate.",
  "The oracle selects the correct branch for each image; the last column is the "
  "fraction of test images on which every branch is already correct."),
 "fig:headroom": ("The ceiling for per-image fusion.",
  "The gap between posterior averaging and the per-image oracle is the entire "
  "budget available to any gating rule."),
 "tab:abl": ("Ablation of the descriptor and the objective.",
  "$\\Delta$ is the change in overall accuracy with respect to the full gate."),
 "tab:eff": ("Parameters, floating-point cost and latency.",
  "Measured per scene on an RTX 2070 Super for UC~Merced at scale factor four."),
 "fig:qge": ("Overall accuracy of the fusion strategies.",
  "Solid bars are the idealised degradation P1, faded bars the unseen "
  "operational degradation P2. Error bars are one standard deviation over three "
  "seeds."),
 "fig:gate": ("Gate weights and the accuracy they select.",
  "Left of each pair: distribution of the gate weights over the test split. "
  "Right of each pair: accuracy of the bicubic branch, the adversarial branch "
  "and the gated mixture within quartiles of the weight assigned to the "
  "adversarial branch."),
 "fig:qual": ("Test scenes with their assigned gate weights.",
  "Weights are $w=(w_{\\mathrm{bic}}, w_{\\mathrm{EDSR}}, w_{\\mathrm{HFGAN}})$. "
  "The upper rows are scenes on which the adversarial branch is wrong and the "
  "gate suppresses it; the lower rows are scenes on which it is right and the "
  "gate promotes it."),
 "fig:pd": ("The perception-distortion plane.",
  "Marker area encodes matched-domain overall accuracy. The model with the best "
  "PSNR is not the model with the best downstream accuracy."),
}

NOTE_OPEN = "\n\\vspace{2pt}\n{\\scriptsize\\raggedright "
NOTE_CLOSE = "\\par}\n"


def main():
    s = io.open(P, encoding="utf-8").read()
    if "--list" in sys.argv:
        for _, j, k, _, lab in spans(s):
            print("%-22s %3d words" % (lab, len(s[j:k].split())))
        return
    # rewrite from the end so earlier offsets stay valid
    for i, j, k, end, lab in reversed(spans(s)):
        if lab not in NEW:
            print("  no rule for", lab)
            continue
        short, note = NEW[lab]
        assert len(short.split()) <= 10, (lab, len(short.split()))
        s = s[:j] + short + s[k:]
        # place the note just before the float closes
        tail = s[end:]
        for closer in ("\\end{table}", "\\end{table*}", "\\end{figure}", "\\end{figure*}"):
            pos = tail.find(closer)
            if pos >= 0 and (pos < 4000):
                ins = end + pos
                s = s[:ins] + NOTE_OPEN + note + NOTE_CLOSE + s[ins:]
                break
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("captions shortened;", len(NEW), "floats annotated")


if __name__ == "__main__":
    main()
