"""LaTeX's \\input is not usable between the rows of a tabular since the file
hooks were introduced, so the generated table bodies are pulled in with the TeX
primitive instead. This patch rewrites main.tex accordingly (idempotent)."""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "paper", "main.tex")
s = io.open(P, encoding="utf-8").read()

macro = ("\\makeatletter\n"
         "\\newcommand{\\inputrows}[1]{\\@@input #1 }\n"
         "\\makeatother\n")
if "\\inputrows" not in s:
    s = s.replace("\\input{auto/numbers}", macro + "\\input{auto/numbers}")

s = re.sub(r"\\input\{auto/(tab_[a-z0-9_]+)\}", r"\\inputrows{auto/\1.tex}", s)
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("patched:", len(re.findall(r"inputrows", s)), "inputrows occurrences")
