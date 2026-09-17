"""ConvNeXt cannot ingest EuroSAT's 16x16 LR-native domain, so the naive gain is
undefined there. Make the confound table tolerate a missing LR-native row."""
import io
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py")
s = io.open(P, encoding="utf-8").read()

old = """        rows[m] = dict(sr=sr, naive_gain=sr[0] - lr_acc[0], controlled_gain=sr[0] - bic_acc[0],
                       oracle_gap=hr_acc[0] - sr[0])"""
new = """        rows[m] = dict(sr=sr,
                       naive_gain=(sr[0] - lr_acc[0]) if lr_acc else None,
                       controlled_gain=sr[0] - bic_acc[0],
                       oracle_gap=hr_acc[0] - sr[0])"""
assert old in s
s = s.replace(old, new)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)

# the LaTeX emitter must print "--" for a missing naive gain
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_tables.py")
s = io.open(P, encoding="utf-8").read()
old = """            rows.append("%s & %s & %+.2f & %+.2f & %.2f & %s \\\\\\\\" %
                        (PRETTY[m], pm(r["sr"]), r["naive_gain"] * 100,
                         r["controlled_gain"] * 100, r["oracle_gap"] * 100,
                         p_str(r.get("mcnemar_p_vs_bic"))))"""
new = """            ng = ("%+.2f" % (r["naive_gain"] * 100)) if r.get("naive_gain") is not None else "--"
            rows.append("%s & %s & %s & %+.2f & %.2f & %s \\\\\\\\" %
                        (PRETTY[m], pm(r["sr"]), ng,
                         r["controlled_gain"] * 100, r["oracle_gap"] * 100,
                         p_str(r.get("mcnemar_p_vs_bic"))))"""
assert old in s, "naive-gain row not found"
s = s.replace(old, new)

old2 = """            mac("naive" + ds + m, "%+.2f" % (r["naive_gain"] * 100))"""
new2 = """            if r.get("naive_gain") is not None:
                mac("naive" + ds + m, "%+.2f" % (r["naive_gain"] * 100))"""
assert old2 in s
s = s.replace(old2, new2)

old3 = """        rows.append("LR-native ($s\\\\times$ smaller input) & %s & -- & -- & -- & -- \\\\\\\\"
                    % pm(cf["lr"]))"""
new3 = """        if cf["lr"]:
            rows.append("LR-native ($s\\\\times$ smaller input) & %s & -- & -- & -- & -- \\\\\\\\"
                        % pm(cf["lr"]))
            mac("LRacc" + ds, "%.2f" % (cf["lr"][0] * 100))"""
assert old3 in s
s = s.replace(old3, new3)
s = s.replace('        mac("LRacc" + ds, "%.2f" % (cf["lr"][0] * 100))\n        mac("BICacc"',
              '        mac("BICacc"')
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("LR-native guards applied")
