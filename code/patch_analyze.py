"""Make analyze.py tolerant of a partially complete run so that one missing
artefact cannot destroy every table."""
import io
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py")
s = io.open(P, encoding="utf-8").read()

old = '''    if accs["bic_p1"] is None or accs[model + "_p1"] is None:
        return None'''
new = '''    if accs["bic_p1"] is None or accs[model + "_p1"] is None:
        return None
    if accs["hr"] is None:
        accs["hr"] = np.full_like(accs["bic_p1"], np.nan)'''
assert old in s
s = s.replace(old, new)

old = '''        d = {}
        d["factorial"] = table_factorial(ds)
        d["confound"] = table_confound(ds)
        d["perclass_hfgan_g"] = table_perclass(ds, "hfgan_g")
        d["perclass_edsr"] = table_perclass(ds, "edsr")
        d["sr_quality"] = table_sr_quality(ds)
        if os.path.exists(os.path.join(C.RESULTS, "srquality_" + ds + ".npz")):
            d["quality_vs_gain"] = table_quality_vs_gain(ds)'''
new = '''        d = {}
        for key, fn in [("factorial", lambda: table_factorial(ds)),
                        ("confound", lambda: table_confound(ds)),
                        ("perclass_hfgan_g", lambda: table_perclass(ds, "hfgan_g")),
                        ("perclass_edsr", lambda: table_perclass(ds, "edsr")),
                        ("sr_quality", lambda: table_sr_quality(ds)),
                        ("quality_vs_gain", lambda: table_quality_vs_gain(ds))]:
            try:
                d[key] = fn()
            except Exception as e:
                print("  SKIP", ds, key, type(e).__name__, e)'''
assert old in s
s = s.replace(old, new)

old = '''    for ds, d in all_out.items():
        print("=" * 70)'''
new = '''    for ds, d in all_out.items():
        if not d.get("confound"):
            continue
        print("=" * 70)'''
assert old in s
s = s.replace(old, new)

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("analyze.py hardened")
