"""Emit every LaTeX table and every inline number macro from all_tables.json.

Nothing in the manuscript is typed by hand: the text uses \\Num{...} macros that
are defined here, so the prose can never drift from the measured results.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

TEX = os.path.join(C.ROOT, "paper", "auto")
os.makedirs(TEX, exist_ok=True)
T = json.load(open(os.path.join(C.RESULTS, "all_tables.json")))
SR_MODELS = ["srcnn", "edsr", "hfgan_p", "hfgan_g"]
PRETTY = {"bic": "Bicubic $\\uparrow$4", "srcnn": "SRCNN", "edsr": "EDSR",
          "hfgan_p": "HFGAN-P", "hfgan_g": "HFGAN-G"}
DSNAME = {"ucmerced": "UC Merced", "eurosat": "EuroSAT"}
MACROS = {}


def mac(name, val):
    MACROS["".join(ch for ch in name if ch.isalnum())] = val


def pm(ms, mult=100.0, nd=2):
    if ms is None:
        return "--"
    return "%.*f\\,$\\pm$\\,%.*f" % (nd, ms[0] * mult, nd, ms[1] * mult)


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
    open(os.path.join(TEX, name + ".tex"), "w", encoding="utf-8").write(body)
    print("  wrote", name + ".tex")


# ------------------------------------------------------- Table: SR quality
def tab_srquality():
    rows = []
    for ds in ["ucmerced", "eurosat"]:
        if ds not in T or not T[ds].get("sr_quality"):
            continue
        q = T[ds]["sr_quality"]
        first = True
        for m in ["bic"] + SR_MODELS:
            k = m + "_p1"
            if k not in q:
                continue
            par = q[k].get("params")
            rows.append("%s & %s & %.2f\\,$\\pm$\\,%.2f & %.4f\\,$\\pm$\\,%.4f & "
                        "%.4f\\,$\\pm$\\,%.4f & %s \\\\" %
                        (DSNAME[ds] if first else "", PRETTY[m],
                         q[k]["psnr"][0], q[k]["psnr"][1],
                         q[k]["ssim"][0], q[k]["ssim"][1],
                         q[k]["lpips"][0], q[k]["lpips"][1],
                         ("%.2f" % (par / 1e6)) if par else "--"))
            mac("PSNR" + ds + m, "%.2f" % q[k]["psnr"][0])
            mac("LPIPS" + ds + m, "%.3f" % q[k]["lpips"][0])
            first = False
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    write("tab_srquality", "\n".join(rows))


# -------------------------------------------------- Table: controlled gain
def tab_confound():
    rows = []
    for ds in ["ucmerced", "eurosat"]:
        if ds not in T:
            continue
        cf = T[ds]["confound"]
        rows.append("\\multicolumn{6}{l}{\\textit{%s}} \\\\" % DSNAME[ds])
        if cf["lr"]:
            rows.append("LR-native ($s\\times$ smaller input) & %s & -- & -- & -- & -- \\\\"
                        % pm(cf["lr"]))
            mac("LRacc" + ds, "%.2f" % (cf["lr"][0] * 100))
        rows.append("Bicubic $\\uparrow$4 (control) & %s & -- & -- & -- & -- \\\\"
                    % pm(cf["bic"]))
        mac("BICacc" + ds, "%.2f" % (cf["bic"][0] * 100))
        mac("HRacc" + ds, "%.2f" % (cf["hr"][0] * 100))
        for m in SR_MODELS:
            if m not in cf["models"]:
                continue
            r = cf["models"][m]
            ng = ("%+.2f" % (r["naive_gain"] * 100)) if r.get("naive_gain") is not None else "--"
            rows.append("%s & %s & %s & %+.2f & %.2f & %s \\\\" %
                        (PRETTY[m], pm(r["sr"]), ng,
                         r["controlled_gain"] * 100, r["oracle_gap"] * 100,
                         p_str(r.get("mcnemar_p_vs_bic"))))
            mac("SRacc" + ds + m, "%.2f" % (r["sr"][0] * 100))
            if r.get("naive_gain") is not None:
                mac("naive" + ds + m, "%+.2f" % (r["naive_gain"] * 100))
            mac("ctrl" + ds + m, "%+.2f" % (r["controlled_gain"] * 100))
            mac("gap" + ds + m, "%.2f" % (r["oracle_gap"] * 100))
            mac("pval" + ds + m, p_str(r.get("mcnemar_p_vs_bic")))
            mac("prpval" + ds + m, p_prose(r.get("mcnemar_p_vs_bic")))
        rows.append("HR reference (oracle) & %s & -- & -- & 0.00 & -- \\\\" % pm(cf["hr"]))
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    write("tab_confound", "\n".join(rows))


# ------------------------------------------------------ Table: correlation
def tab_corr():
    rows = []
    for ds in ["ucmerced", "eurosat"]:
        if ds not in T or "quality_vs_gain" in T[ds] is False:
            continue
        qv = T[ds].get("quality_vs_gain")
        if not qv:
            continue
        first = True
        for k, lab in [("d_psnr", "$\\Delta$PSNR (dB)"), ("d_ssim", "$\\Delta$SSIM"),
                       ("d_lpips", "$\\Delta$LPIPS")]:
            v = qv[k]
            rows.append("%s & %s & %+.3f & %s & %+.3f & %s & %d \\\\" %
                        (DSNAME[ds] if first else "", lab, v["spearman"],
                         p_str(v["spearman_p"]), v["pearson"], p_str(v["pearson_p"]),
                         v["n"]))
            mac("rho" + ds + k.replace("_", ""), "%+.3f" % v["spearman"])
            mac("rhop" + ds + k.replace("_", ""), p_str(v["spearman_p"]))
            mac("prrhop" + ds + k.replace("_", ""), p_prose(v["spearman_p"]))
            first = False
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    write("tab_corr", "\n".join(rows))


# ------------------------------------------------------------ Table: perclass
def tab_perclass(ds="ucmerced"):
    d = T.get(ds, {}).get("perclass_hfgan_g")
    if not d:
        return
    o = np.array(d["order"])
    rows = []
    for i in o:
        rows.append("%s & %.1f & %.1f & %+.1f & %.1f \\\\" %
                    (d["classes"][i].replace("_", " "), d["bic"][i] * 100,
                     d["sr"][i] * 100, d["delta"][i] * 100, d["hr"][i] * 100))
    delta = np.array(d["delta"])
    mac("nClsUp", str(int((delta > 0).sum())))
    mac("nClsDown", str(int((delta < 0).sum())))
    mac("nClsFlat", str(int((delta == 0).sum())))
    mac("bestCls", d["classes"][int(o[0])].replace("_", " "))
    mac("bestClsGain", "%+.1f" % (delta[o[0]] * 100))
    mac("worstCls", d["classes"][int(o[-1])].replace("_", " "))
    mac("worstClsGain", "%+.1f" % (delta[o[-1]] * 100))
    write("tab_perclass", "\n".join(rows))


# ----------------------------------------------------------- Table: QGE
QGE_KEYS = [("single_bic", "Bicubic branch alone"),
            ("single_edsr", "EDSR branch alone"),
            ("single_hfgan_g", "HFGAN-G branch alone"),
            ("majority", "Majority vote"),
            ("avg", "Posterior averaging"),
            ("global_w", "Global weighted vote"),
            ("stacking", "Stacking (logistic)"),
            ("confmax", "Confidence-max routing"),
            ("arch_ens", "Architecture ensemble (3 CNNs)"),
            ("qge", "Quality-gated ensemble")]


def _agg(blk, k, field):
    v = [blk[s][k][field] for s in blk]
    return float(np.mean(v)), float(np.std(v))


def tab_qge(protocol="p1"):
    rows = []
    for ds in ["ucmerced", "eurosat"]:
        blk = T.get(ds, {}).get("qge_" + protocol)
        if not blk:
            continue
        rows.append("\\multicolumn{5}{l}{\\textit{%s}} \\\\" % DSNAME[ds])
        for k, lab in QGE_KEYS:
            if k not in blk[list(blk)[0]]:
                continue
            rows.append("%s & %s & %s & %s & %.3f \\\\" %
                        (lab, pm(_agg(blk, k, "oa")), pm(_agg(blk, k, "mf1")),
                         pm(_agg(blk, k, "kappa")), _agg(blk, k, "ece")[0]))
            mac("qge" + protocol + ds + k.replace("_", ""),
                "%.2f" % (_agg(blk, k, "oa")[0] * 100))
        mcn = [blk[s]["mcnemar_vs_best_baseline"] for s in blk]
        pmed = float(np.median([m["p"] for m in mcn]))
        mac("qgeP" + protocol + ds, p_str(pmed))
        mac("prqgeP" + protocol + ds, p_prose(pmed))
        mac("qgeBase" + protocol + ds, mcn[0]["baseline"].replace("_", "-"))
        w = np.mean([blk[s]["qge_mean_weights"] for s in blk], 0)
        mac("qgeW" + protocol + ds, "(%.2f, %.2f, %.2f)" % tuple(w))
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    write("tab_qge_" + protocol, "\n".join(rows))


def tab_ablation():
    labels = {"no_realism": "w/o discriminator realism features",
              "no_crossview": "w/o cross-view residual features",
              "no_posterior": "w/o branch-posterior features",
              "no_branch_sup": "w/o branch supervision term ($\\alpha=0$)",
              "no_halluc_pen": "w/o hallucination penalty ($\\gamma=0$)"}
    rows = []
    for ds in ["ucmerced", "eurosat"]:
        full = T.get(ds, {}).get("qge_p1")
        if not full:
            continue
        base = _agg(full, "qge", "oa")
        rows.append("\\multicolumn{4}{l}{\\textit{%s}} \\\\" % DSNAME[ds])
        rows.append("Full QGE & %s & %s & -- \\\\" %
                    (pm(base), pm(_agg(full, "qge", "mf1"))))
        for ab, lab in labels.items():
            blk = T[ds].get("qge_ablation_" + ab)
            if not blk:
                continue
            a = _agg(blk, "qge", "oa")
            rows.append("%s & %s & %s & %+.2f \\\\" %
                        (lab, pm(a), pm(_agg(blk, "qge", "mf1")),
                         (a[0] - base[0]) * 100))
            mac("abl" + ds + ab.replace("_", ""), "%+.2f" % ((a[0] - base[0]) * 100))
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    write("tab_ablation", "\n".join(rows))


def tab_efficiency():
    f = os.path.join(C.RESULTS, "efficiency.json")
    if not os.path.exists(f):
        return
    e = json.load(open(f))
    rows = []
    for k, v in e.items():
        rows.append("%s & %.2f & %.2f & %.2f \\\\" %
                    (v["label"], v["params_m"], v["gflops"], v["ms"]))
        mac("ms" + k.replace("_", ""), "%.2f" % v["ms"])
    write("tab_efficiency", "\n".join(rows))


def main():
    tab_srquality()
    tab_confound()
    tab_corr()
    tab_perclass()
    tab_qge("p1")
    tab_qge("p2")
    tab_ablation()
    tab_efficiency()
    lines = []
    for k, v in sorted(MACROS.items()):
        lines.append("\\expandafter\\def\\csname NUM%s\\endcsname{%s}" % (k, v))
    lines.append("\\newcommand{\\N}[1]{\\csname NUM#1\\endcsname}")
    open(os.path.join(TEX, "numbers.tex"), "w", encoding="utf-8").write("\n".join(lines))
    print("  wrote numbers.tex with", len(MACROS), "macros")
    json.dump(MACROS, open(os.path.join(C.RESULTS, "macros.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
