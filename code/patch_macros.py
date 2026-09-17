"""Macro names must keep their digits, otherwise the P1 and P2 protocol macros
collide. TeX allows digits inside a name built with \\csname."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))

p = os.path.join(HERE, "make_tables.py")
s = io.open(p, encoding="utf-8").read()
s = s.replace('MACROS["".join(ch for ch in name if ch.isalpha())] = val',
              'MACROS["".join(ch for ch in name if ch.isalnum())] = val')
s = s.replace('        safe = "".join(ch for ch in k if ch.isalpha())\n', '')
s = s.replace('% (safe, v))', '% (k, v))')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

p = os.path.join(HERE, "..", "paper", "main.tex")
s = io.open(p, encoding="utf-8").read()
s = s.replace("\\N{qgePpucmerced}", "\\N{qgePp1ucmerced}")
s = s.replace("\\N{qgePpeurosat}", "\\N{qgePp1eurosat}")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("macro naming patched")
