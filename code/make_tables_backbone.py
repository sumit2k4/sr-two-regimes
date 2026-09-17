"""Backbone-robustness table: every headline quantity computed twice, once with
ConvNeXt-Tiny branches and once with ResNet-18 branches, from the same cached
reconstructions and the same fixed split."""
import io
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import analyze
import analyze_transfer as AT

TEX = os.path.join(C.ROOT, "paper", "auto")
SR_MODELS = ["srcnn", "edsr", "hfgan_p", "hfgan_g"]
DSNAME = {"ucmerced": "UC Merced", "eurosat": "EuroSAT"}
MACROS = {}


def mac(name, val):
    MACROS["".join(ch for ch in name if ch.isalnum())] = val


def collect(backbone):
    out = {}
    for ds in ["ucmerced", "eurosat"]:
        cf = analyze.table_confound(ds, backbone)
        qv = analyze.table_quality_vs_gain(ds, backbone)
        AT.BACKBONE = backbone
        tr = AT.transfer_block(ds)
        tc = AT.transfer_corr(ds)
        ctrl = [r["controlled_gain"] * 100 for r in cf["models"].values()]
        gains = [tr[m]["gain"] * 100 for m in SR_MODELS if m in tr]
        sig_pos = sum(1 for m, r in cf["models"].items()
                      if r["controlled_gain"] > 0
                      and (r.get("mcnemar_p_vs_bic") or 1) < 0.05)
        sig_neg = sum(1 for m, r in cf["models"].items()
                      if r["controlled_gain"] < 0
                      and (r.get("mcnemar_p_vs_bic") or 1) < 0.05)
        out[ds] = dict(hr=cf["hr"][0] * 100, bic=cf["bic"][0] * 100,
                       ctrl_lo=min(ctrl), ctrl_hi=max(ctrl),
                       sig_pos=sig_pos, sig_neg=sig_neg,
                       tr_lo=min(gains), tr_hi=max(gains),
                       rho_psnr=qv["d_psnr"]["spearman"],
                       rho_lpips_tr=tc["d_lpips"]["spearman"],
                       p_lpips_tr=tc["d_lpips"]["spearman_p"])
    return out


def main():
    A = collect("convnext_tiny")
    B = collect("resnet18")
    rows = []
    LBL = [("hr", "HR reference OA (\\%)", "%.2f"),
           ("bic", "Bicubic control OA (\\%)", "%.2f"),
           ("ctrl_lo", "Matched controlled gain, min (pp)", "%+.2f"),
           ("ctrl_hi", "Matched controlled gain, max (pp)", "%+.2f"),
           ("sig_pos", "Significant \\emph{positive} controlled gains", "%d"),
           ("sig_neg", "Significant \\emph{negative} controlled gains", "%d"),
           ("tr_lo", "Transfer gain, min (pp)", "%+.2f"),
           ("tr_hi", "Transfer gain, max (pp)", "%+.2f"),
           ("rho_psnr", "Matched: $\\rho(\\Delta$PSNR$,\\Delta$OA$)$", "%+.3f"),
           ("rho_lpips_tr", "Transfer: $\\rho(\\Delta$LPIPS$,\\Delta$OA$)$", "%+.3f")]
    for ds in ["ucmerced", "eurosat"]:
        rows.append("\\multicolumn{3}{l}{\\textit{%s}} \\\\" % DSNAME[ds])
        for k, lab, fmt in LBL:
            rows.append("%s & %s & %s \\\\" % (lab, fmt % A[ds][k], fmt % B[ds][k]))
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    io.open(os.path.join(TEX, "tab_backbone.tex"), "w", encoding="utf-8",
            newline="\n").write("\n".join(rows))

    for ds in ["ucmerced", "eurosat"]:
        for tag, D in [("cnx", A), ("r18", B)]:
            mac("bb" + tag + ds + "hr", "%.2f" % D[ds]["hr"])
            mac("bb" + tag + ds + "trhi", "%+.2f" % D[ds]["tr_hi"])
            mac("bb" + tag + ds + "ctrlhi", "%+.2f" % D[ds]["ctrl_hi"])
    mac("bbSigPos", str(sum(A[d]["sig_pos"] + B[d]["sig_pos"]
                            for d in ["ucmerced", "eurosat"])))

    # saturation arithmetic for UC Merced
    sp = C.get_splits("ucmerced")
    n = len(sp["test"])
    mac("ucmTestN", str(n))
    mac("ucmOneErr", "%.2f" % (100.0 * (1 - 1.0 / n)))
    mac("ucmTwoErr", "%.2f" % (100.0 * (1 - 2.0 / n)))
    mac("ucmOurErr", "%d" % round(n * (1 - A["ucmerced"]["hr"] / 100.0)))

    p = os.path.join(TEX, "numbers.tex")
    existing = io.open(p, encoding="utf-8").read()
    lines = [l for l in existing.split("\n")
             if l.strip() and l.startswith("\\expandafter")]
    for k, v in sorted(MACROS.items()):
        lines.append("\\expandafter\\def\\csname NUM%s\\endcsname{%s}" % (k, v))
    lines.append("\\makeatletter")
    lines.append("\\def\\N#1{\\@ifundefined{NUM#1}{\\textbf{??}}{\\csname NUM#1\\endcsname}}")
    lines.append("\\makeatother")
    io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    print("tab_backbone.tex written;", len(MACROS), "macros appended")
    for ds in ["ucmerced", "eurosat"]:
        print(ds, "ConvNeXt HR %.2f  R18 HR %.2f | ctrl %.2f..%.2f vs %.2f..%.2f | "
              "transfer max %+.2f vs %+.2f"
              % (A[ds]["hr"], B[ds]["hr"], A[ds]["ctrl_lo"], A[ds]["ctrl_hi"],
                 B[ds]["ctrl_lo"], B[ds]["ctrl_hi"], A[ds]["tr_hi"], B[ds]["tr_hi"]))


if __name__ == "__main__":
    main()
