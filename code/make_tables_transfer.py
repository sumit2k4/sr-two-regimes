"""LaTeX tables and macros for the transfer regime and the gate-headroom
analysis. Appends to the macro file written by make_tables.py."""
import io
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

TEX = os.path.join(C.ROOT, "paper", "auto")
os.makedirs(TEX, exist_ok=True)
PRETTY = {"bic": "Bicubic $\\uparrow$4 (control)", "srcnn": "SRCNN", "edsr": "EDSR",
          "hfgan_p": "HFGAN-P", "hfgan_g": "HFGAN-G"}
DSNAME = {"ucmerced": "UC Merced", "eurosat": "EuroSAT"}
SR_MODELS = ["srcnn", "edsr", "hfgan_p", "hfgan_g"]
MACROS = {}


def mac(name, val):
    MACROS["".join(ch for ch in name if ch.isalnum())] = val


def p_str(p):
    if p is None:
        return "--"
    if p < 1e-4:
        return "\\ensuremath{<}10^{-4}" if False else "\\ensuremath{<10^{-4}}"
    return "%.3g" % p


def p_prose(p):
    """Math-mode body carrying the relation, for use as $p\\N{pr...}$."""
    if p is None:
        return "{}"
    if p < 1e-4:
        return "{<}10^{-4}"
    return "{=}%.3g" % p

def write(name, body):
    io.open(os.path.join(TEX, name + ".tex"), "w", encoding="utf-8",
            newline="\n").write(body)
    print("  wrote", name + ".tex")


def tab_transfer(T):
    rows = []
    for ds in ["ucmerced", "eurosat"]:
        if ds not in T:
            continue
        b = T[ds]["transfer"]
        rows.append("\\multicolumn{4}{l}{\\textit{%s}\\quad(HR-trained classifier, "
                    "matched HR accuracy %.2f\\,\\%%)} \\\\"
                    % (DSNAME[ds], b["_hr_matched"][0] * 100))
        mac("trHR" + ds, "%.2f" % (b["_hr_matched"][0] * 100))
        for m in ["bic"] + SR_MODELS:
            if m not in b:
                continue
            r = b[m]
            rows.append("%s & %.2f\\,$\\pm$\\,%.2f & %s & %s \\\\" %
                        (PRETTY[m], r["acc"][0] * 100, r["acc"][1] * 100,
                         ("--" if m == "bic" else "%+.2f" % (r["gain"] * 100)),
                         p_str(r.get("p"))))
            mac("tracc" + ds + m, "%.2f" % (r["acc"][0] * 100))
            if m != "bic":
                mac("trgain" + ds + m, "%+.2f" % (r["gain"] * 100))
                mac("trp" + ds + m, p_str(r.get("p")))
                mac("prtrp" + ds + m, p_prose(r.get("p")))
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    write("tab_transfer", "\n".join(rows))


def tab_transfer_corr(T):
    rows = []
    for ds in ["ucmerced", "eurosat"]:
        if ds not in T:
            continue
        qv = T[ds]["transfer_corr"]
        first = True
        for k, lab in [("d_psnr", "$\\Delta$PSNR (dB)"), ("d_ssim", "$\\Delta$SSIM"),
                       ("d_lpips", "$\\Delta$LPIPS")]:
            v = qv[k]
            rows.append("%s & %s & %+.3f & %s & %d \\\\" %
                        (DSNAME[ds] if first else "", lab, v["spearman"],
                         p_str(v["spearman_p"]), v["n"]))
            mac("trrho" + ds + k.replace("_", ""), "%+.3f" % v["spearman"])
            mac("trrhop" + ds + k.replace("_", ""), p_str(v["spearman_p"]))
            mac("prtrrhop" + ds + k.replace("_", ""), p_prose(v["spearman_p"]))
            first = False
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    write("tab_transfer_corr", "\n".join(rows))


def tab_headroom():
    f = os.path.join(C.RESULTS, "gate_headroom.json")
    if not os.path.exists(f):
        return
    H = json.load(open(f))
    rows = []
    for key in ["ucmerced_p1", "eurosat_p1"]:
        if key not in H:
            continue
        ds = key.split("_")[0]
        for tag, lab in [("views", "3 views, ResNet-18"),
                         ("extended", "5 experts (views $\\times$ arch.)")]:
            if tag not in H[key]:
                continue
            d = H[key][tag]
            rows.append("%s & %s & %.2f & %.2f & %.2f & %.1f \\\\" %
                        (DSNAME[ds] if tag == "views" else "", lab,
                         d["best_single"][0] * 100, d["avg"][0] * 100,
                         d["oracle_per_image"][0] * 100,
                         d["all_branches_correct"][0] * 100))
            mac("hd" + ds + tag + "avg", "%.2f" % (d["avg"][0] * 100))
            mac("hd" + ds + tag + "oracle", "%.2f" % (d["oracle_per_image"][0] * 100))
            mac("hd" + ds + tag + "agree", "%.1f" % (d["all_branches_correct"][0] * 100))
            mac("hd" + ds + tag + "head",
                "%.2f" % ((d["oracle_per_image"][0] - d["avg"][0]) * 100))
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    write("tab_headroom", "\n".join(rows))


def main():
    T = json.load(open(os.path.join(C.RESULTS, "transfer.json")))
    tab_transfer(T)
    tab_transfer_corr(T)
    tab_headroom()
    # append to the macro file produced by make_tables.py
    p = os.path.join(TEX, "numbers.tex")
    existing = io.open(p, encoding="utf-8").read() if os.path.exists(p) else ""
    lines = [l for l in existing.split("\n")
             if l.strip() and l.startswith("\\expandafter")]
    for k, v in sorted(MACROS.items()):
        lines.append("\\expandafter\\def\\csname NUM%s\\endcsname{%s}" % (k, v))
    lines.append("\\makeatletter")
    lines.append("\\def\\N#1{\\@ifundefined{NUM#1}{\\textbf{??}}{\\csname NUM#1\\endcsname}}")
    lines.append("\\makeatother")
    io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    print("  numbers.tex now has", len(lines) - 3, "macros")


if __name__ == "__main__":
    main()
