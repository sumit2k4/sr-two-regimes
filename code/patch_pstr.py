"""p-value macros are used both inside table cells (text mode) and inside
$...$ in the prose. Emit a form that is legal in both."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_A = '''    if p < 1e-4:
        return "$<$1e-4"
    return "%.3g" % p'''
OLD_B = '''    if p < 1e-4:
        return "$<10^{-4}$"
    return "%.3g" % p'''
NEW = '''    if p < 1e-4:
        return "\\\\ensuremath{<}10^{-4}" if False else "\\\\ensuremath{<10^{-4}}"
    return "%.3g" % p'''

for name in ("make_tables.py", "make_tables_transfer.py"):
    p = os.path.join(HERE, name)
    s = io.open(p, encoding="utf-8").read()
    for old in (OLD_A, OLD_B):
        if old in s:
            s = s.replace(old, NEW)
            io.open(p, "w", encoding="utf-8", newline="\n").write(s)
            print("patched", name)
            break
    else:
        print("no match in", name)
