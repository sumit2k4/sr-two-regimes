"""Three layout fixes found by inspecting the compiled PDF:
  1. the matched-regime table overflows its column and collides with a neighbour
  2. "$p=\\N{...}$" renders as "p =< 1e-4" when the macro carries a relation
  3. the four-panel correlation figure is unreadable at single-column width
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "paper", "main.tex")
s = io.open(P, encoding="utf-8").read()

# --- 1. narrow the matched-regime table -------------------------------------
old = """\\centering
\\footnotesize
\\begin{tabular}{lccccc}
\\toprule
Input domain & OA (\\%) & $\\Delta_{\\mathrm{naive}}$ & $\\Delta_{\\mathrm{ctrl}}$
& $\\Delta_{\\mathrm{oracle}}$ & $p$ \\\\"""
new = """\\centering
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\begin{tabular}{lccccc}
\\toprule
Input domain & OA (\\%) & $\\Delta_{\\mathrm{naive}}$ & $\\Delta_{\\mathrm{ctrl}}$
& $\\Delta_{\\mathrm{oracle}}$ & $p$ \\\\"""
assert old in s
s = s.replace(old, new)

# --- also narrow the two correlation tables and the fusion tables ------------
for spec in ["{llccccc}", "{llccc}", "{lcccc}", "{llcccc}", "{lccc}"]:
    s = s.replace("\\footnotesize\n\\begin{tabular}" + spec,
                  "\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n"
                  "\\begin{tabular}" + spec)

# --- 2. prose p-values ------------------------------------------------------
# the macros carry only the value; supply the relation from a prose-safe macro
s = re.sub(r"\$p=\\N\{(trp[A-Za-z0-9]+)\}\$", r"$p$\\,\\N{\1}", s)
s = re.sub(r"\$p=\\N\{(pval[A-Za-z0-9]+)\}\$", r"$p$\\,\\N{\1}", s)
s = re.sub(r"\$p=\\N\{(rhop[A-Za-z0-9]+)\}\$", r"$p$\\,\\N{\1}", s)
s = re.sub(r"\$p=\\N\{(trrhop[A-Za-z0-9]+)\}\$", r"$p$\\,\\N{\1}", s)
s = s.replace("$p=\\N{qgePp1ucmerced}$", "$p$\\,\\N{qgePp1ucmerced}")
s = s.replace("$p=\\N{qgePp1eurosat}$", "$p$\\,\\N{qgePp1eurosat}")

# --- 3. widen the correlation figure ----------------------------------------
old = """\\begin{figure}[!t]
\\centering
\\includegraphics[width=\\columnwidth]{fig_lpips_transfer.pdf}"""
new = """\\begin{figure*}[!t]
\\centering
\\includegraphics[width=\\textwidth]{fig_lpips_transfer.pdf}"""
assert old in s
s = s.replace(old, new)
s = s.replace("""distance carries no
information about the matched-regime outcome and predicts the transfer outcome.}
\\label{fig:lpipstransfer}
\\end{figure}""",
              """distance carries no
information about the matched-regime outcome and predicts the transfer outcome.}
\\label{fig:lpipstransfer}
\\end{figure*}""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("layout patched")
