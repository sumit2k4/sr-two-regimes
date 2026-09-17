"""Tables VII and VIII overflow the column and collide with their neighbours.
Shorten the row labels and the headers rather than shrinking the type further."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- shorter row labels in the fusion table --------------------------------
p = os.path.join(HERE, "make_tables.py")
s = io.open(p, encoding="utf-8").read()
s = s.replace('("arch_ens", "Architecture ensemble (3 CNNs, SR view)")',
              '("arch_ens", "Architecture ensemble (3 CNNs)")')
s = s.replace('("qge", "Quality-gated ensemble (tested here)")',
              '("qge", "Quality-gated ensemble")')
s = s.replace('("confmax", "Confidence-max routing")',
              '("confmax", "Confidence-max routing")')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

# ---- shorter expert-set labels in the headroom table -----------------------
p = os.path.join(HERE, "make_tables_transfer.py")
s = io.open(p, encoding="utf-8").read()
s = s.replace('("views", "3 views, ResNet-18"),\n'
              '                         ("extended", "3 views $\\\\times$ architectures (5 experts)")',
              '("views", "3 views, ResNet-18"),\n'
              '                         ("extended", "5 experts (views $\\\\times$ arch.)")')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

# ---- shorter headers in the manuscript -------------------------------------
p = os.path.join(HERE, "..", "paper", "main.tex")
s = io.open(p, encoding="utf-8").read()
s = s.replace("Method & OA (\\%) & macro $F_1$ (\\%) & $\\kappa$ ($\\times100$) & ECE \\\\",
              "Method & OA (\\%) & m$F_1$ (\\%) & $\\kappa$ & ECE \\\\")
s = s.replace("Dataset & Expert set & Best single & Average & Oracle & All correct (\\%) \\\\",
              "Dataset & Expert set & Best & Avg. & Oracle & Agree (\\%) \\\\")
s = s.replace("Variant & OA (\\%) & macro $F_1$ (\\%) & $\\Delta$OA \\\\",
              "Variant & OA (\\%) & m$F_1$ (\\%) & $\\Delta$OA \\\\")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("wide tables narrowed")
