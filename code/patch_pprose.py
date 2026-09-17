"""Give p-values a prose form that carries its own relation symbol, so the text
reads "p = 0.0357" and "p < 10^{-4}" rather than "p 0.0357"."""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

PROSE_FN = '''

def p_prose(p):
    """Math-mode body carrying the relation, for use as $p\\\\N{pr...}$."""
    if p is None:
        return "{}"
    if p < 1e-4:
        return "{<}10^{-4}"
    return "{=}%.3g" % p

'''

for name in ("make_tables.py", "make_tables_transfer.py"):
    p = os.path.join(HERE, name)
    s = io.open(p, encoding="utf-8").read()
    if "def p_prose" not in s:
        anchor = "def write(name, body):"
        assert anchor in s, name
        s = s.replace(anchor, PROSE_FN.strip("\n") + "\n\n" + anchor, 1)
    # every mac(...) that stores a p_str value also stores a prose twin
    s = re.sub(r'mac\((\"[^\"]+\"\s*\+\s*[^,]+|\"[A-Za-z0-9]+\"), p_str\(([^\n]+?)\)\)',
               lambda m: ('mac(%s, p_str(%s))\n            mac("pr" + %s, p_prose(%s))'
                          % (m.group(1), m.group(2), m.group(1), m.group(2))),
               s)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("patched", name, "| prose macros:", s.count("p_prose("))

# --- rewrite the prose in the manuscript ------------------------------------
P = os.path.join(HERE, "..", "paper", "main.tex")
s = io.open(P, encoding="utf-8").read()
s = re.sub(r"\$p\$\\,\\N\{([A-Za-z0-9]+)\}", r"$p\\N{pr\1}$", s)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("main.tex prose p-values:", len(re.findall(r"\\N\{pr[A-Za-z0-9]+\}", s)))

# --- shorten a cramped axis label -------------------------------------------
F = os.path.join(HERE, "figures.py")
s = io.open(F, encoding="utf-8").read()
s = s.replace('ax.set_ylabel("per-class OA gain of HFGAN-G\\nover bicubic control (pp)")',
              'ax.set_ylabel("OA gain over\\nbicubic control (pp)", fontsize=7)')
io.open(F, "w", encoding="utf-8", newline="\n").write(s)
print("figure label shortened")
