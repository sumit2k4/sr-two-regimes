"""Re-place the caption notes inside the correct float.

The first pass chose the first closer found in a fixed list rather than the
nearest one in the text, so a figure whose float ended before a nearby table
had its note pushed into that table. Strip every inserted note and redo it,
choosing the closer with the smallest offset.
"""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "main.tex")
OPEN = "\n\\vspace{2pt}\n{\\scriptsize\\raggedright "
CLOSE = "\\par}\n"

import shorten_captions as SC

s = io.open(P, encoding="utf-8").read()

# --- strip every previously inserted note ----------------------------------
removed = 0
while True:
    i = s.find(OPEN)
    if i < 0:
        break
    j = s.find(CLOSE, i)
    assert j > 0, "unterminated note"
    s = s[:i] + s[j + len(CLOSE):]
    removed += 1
print("stripped", removed, "notes")

# --- re-insert, nearest closer wins ----------------------------------------
CLOSERS = ["\\end{table}", "\\end{table*}", "\\end{figure}", "\\end{figure*}"]
placed = 0
for i, j, k, end, lab in reversed(SC.spans(s)):
    if lab not in SC.NEW:
        continue
    note = SC.NEW[lab][1]
    tail = s[end:end + 6000]
    best, bestpos = None, None
    for c in CLOSERS:
        p = tail.find(c)
        if p >= 0 and (bestpos is None or p < bestpos):
            best, bestpos = c, p
    if best is None:
        print("  no closer for", lab)
        continue
    ins = end + bestpos
    s = s[:ins] + OPEN + note + CLOSE + s[ins:]
    placed += 1

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("re-placed", placed, "notes")
