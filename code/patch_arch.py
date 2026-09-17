"""Layout fixes for the architecture figure found by inspecting the render."""
import io
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_arch.py")
s = io.open(P, encoding="utf-8").read()

# 1. LR imagery drawn smaller than HR, so the size difference is visible
s = s.replace('place(ax, thumb("lr_p1"), 12, 24.0, 11)',
              'place(ax, thumb("lr_p1"), 12, 24.0, 7.0)')
s = s.replace('place(ax, thumb("lr_p1"), 29.4, 12.0, 4.4, lw=0.5)',
              'place(ax, thumb("lr_p1"), 29.4, 12.0, 2.6, lw=0.5)')
s = s.replace('txt(ax, 12, 16.8, "LR observation", fs=6.5, w="bold")',
              'txt(ax, 12, 18.8, "LR observation", fs=6.5, w="bold")')
s = s.replace('arrow(ax, (12, 33.4), (12, 29.7))',
              'arrow(ax, (12, 33.4), (12, 27.8))')

# 2. regime boxes: glyph on the left, text block on the right, no overlap
OLD_M = '''    box(ax, 53.0, 73.5, 41.5, 58.5, GREEN)
    net_icon(ax, 58.6, 52.0, 9)
    txt(ax, 63.2, 56.2, "Matched regime", fs=7.0, w="bold")
    txt(ax, 66.0, 48.6, "a classifier is trained on", fs=5.8)
    txt(ax, 66.0, 46.6, "each domain and tested on", fs=5.8)
    txt(ax, 66.0, 44.6, "that same domain", fs=5.8)
    txt(ax, 66.0, 42.6, "rebuilding the pipeline", fs=5.5, st="italic", c="#5A5A5A")'''
NEW_M = '''    box(ax, 53.0, 73.5, 41.5, 58.5, GREEN)
    txt(ax, 63.2, 56.4, "Matched regime", fs=7.0, w="bold")
    net_icon(ax, 58.4, 48.6, 7.6)
    txt(ax, 62.6, 51.6, "a classifier is trained", fs=5.7, ha="left")
    txt(ax, 62.6, 49.6, "on each domain and", fs=5.7, ha="left")
    txt(ax, 62.6, 47.6, "tested on that domain", fs=5.7, ha="left")
    txt(ax, 63.2, 43.4, "rebuilding the pipeline", fs=5.5, st="italic", c="#5A5A5A")'''
assert OLD_M in s
s = s.replace(OLD_M, NEW_M)

OLD_T = '''    box(ax, 53.0, 73.5, 20.5, 37.5, ORANGE)
    net_icon(ax, 58.6, 30.5, 9)
    txt(ax, 63.2, 35.2, "Transfer regime", fs=7.0, w="bold")
    txt(ax, 66.0, 27.6, "one classifier trained on HR", fs=5.8)
    txt(ax, 66.0, 25.6, "imagery is reused unchanged", fs=5.8)
    txt(ax, 66.0, 23.6, "on every domain", fs=5.8)
    txt(ax, 66.0, 21.6, "reusing an existing model", fs=5.5, st="italic", c="#5A5A5A")'''
NEW_T = '''    box(ax, 53.0, 73.5, 20.5, 37.5, ORANGE)
    txt(ax, 63.2, 35.4, "Transfer regime", fs=7.0, w="bold")
    net_icon(ax, 58.4, 27.6, 7.6)
    txt(ax, 62.6, 30.6, "one HR-trained model", fs=5.7, ha="left")
    txt(ax, 62.6, 28.6, "is reused unchanged", fs=5.7, ha="left")
    txt(ax, 62.6, 26.6, "on every domain", fs=5.7, ha="left")
    txt(ax, 63.2, 22.4, "reusing an existing model", fs=5.5, st="italic", c="#5A5A5A")'''
assert OLD_T in s
s = s.replace(OLD_T, NEW_T)

# 3. drop the stray label that collided with the banner row
s = s.replace('''    txt(ax, 50.6, 61.2, "every domain enters both regimes", fs=5.5, st="italic",
        c="#5A5A5A")
''', "")

# 4. thin the connector bundle so it reads as a bus rather than a blot
s = s.replace('def elbow(ax, x0, y0, x1, y1, col=EC, ls="-", lw=0.7, z=3):',
              'def elbow(ax, x0, y0, x1, y1, col=EC, ls="-", lw=0.55, z=3):')

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("architecture layout patched")
