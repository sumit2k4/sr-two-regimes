"""Verify that every \\N{...} used in the manuscript resolves to a defined
macro, so no '??' placeholder can reach the PDF unnoticed."""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper")

tex = io.open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
nums = io.open(os.path.join(PAPER, "auto", "numbers.tex"), encoding="utf-8").read()

defined = set(re.findall(r"NUM([A-Za-z0-9]+)" + re.escape("\\endcsname"), nums))
used = set(re.findall(re.escape("\\N") + r"\{([A-Za-z0-9]+)\}", tex))

print("macros used in text :", len(used))
print("macros defined      :", len(defined))
missing = sorted(used - defined)
if missing:
    print("MISSING:", missing)
else:
    print("all resolved")
