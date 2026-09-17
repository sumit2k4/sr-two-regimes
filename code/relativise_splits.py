"""Store split files with paths relative to the project root.

The cached split files were written with absolute paths from the machine that
produced them, which is both unportable and inappropriate for a public release.
This rewrites them in place, preserving the train/validation/test index arrays
exactly, so the partitions are unchanged.
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def to_rel(p):
    p = p.replace("\\", "/")
    i = p.lower().find("/data/")
    if i >= 0:
        return p[i + 1:]          # -> data/<dataset>/<class>/<file>.png
    return os.path.basename(p)


for ds in ["ucmerced", "eurosat"]:
    f = os.path.join(C.RESULTS, "splits_" + ds + ".json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    before = (len(d["train"]), len(d["val"]), len(d["test"]))
    d["paths"] = [to_rel(p) for p in d["paths"]]
    after = (len(d["train"]), len(d["val"]), len(d["test"]))
    assert before == after
    io.open(f, "w", encoding="utf-8", newline="\n").write(json.dumps(d))
    print("%-9s %d paths relativised, splits %s unchanged, e.g. %s"
          % (ds, len(d["paths"]), before, d["paths"][0]))
