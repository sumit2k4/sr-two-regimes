"""Expand every abbreviated venue name, including those that wrap across source
lines and those the first pass half-replaced.

Bibliography entries are line-wrapped, so a flat string match misses any name
broken by a newline. Collapse whitespace inside each entry first, then replace.
"""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "main.tex")

FULL = {
 "IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing": [
   "IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.",
   "IEEE J. Sel. Topics Appl. Earth Observ. Remote Sensing",
   "IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sens.",
 ],
 "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops": [
   "Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops",
   "Proceedings of the IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops"],
 "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops": [
   "Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops",
   "Proceedings of the IEEE Conf. Comput. Vis. Pattern Recognit. Workshops"],
 "Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops": [
   "Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops",
   "Proceedings of the IEEE/CVF Int. Conf. Comput. Vis. Workshops"],
 "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition": [
   "Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.",
   "Proceedings of the IEEE/CVF Conf. Comput. Vis. Pattern Recognit."],
 "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition": [
   "Proc. IEEE Conf. Comput. Vis. Pattern Recognit.",
   "Proceedings of the IEEE Conf. Comput. Vis. Pattern Recognit."],
 "Proceedings of the ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems": [
   "Proc. ACM SIGSPATIAL Int. Conf. Adv. Geogr. Inf. Syst."],
 "Proceedings of the IEEE Winter Conference on Applications of Computer Vision": [
   "Proc. IEEE Winter Conf. Appl. Comput. Vis."],
 "Proceedings of the European Conference on Computer Vision Workshops": [
   "Proc. Eur. Conf. Comput. Vis. Workshops"],
 "Proceedings of the International Conference on Learning Representations": [
   "Proc. Int. Conf. Learn. Represent."],
 "Proceedings of the International Conference on Machine Learning": [
   "Proc. Int. Conf. Mach. Learn."],
 "Proceedings of the European Conference on Computer Vision": [
   "Proc. Eur. Conf. Comput. Vis."],
 "ISPRS Journal of Photogrammetry and Remote Sensing": [
   "ISPRS J. Photogramm. Remote Sens.", "ISPRS J. Photogramm. Remote Sensing"],
 "IEEE Transactions on Pattern Analysis and Machine Intelligence": [
   "IEEE Trans. Pattern Anal. Mach. Intell."],
 "International Journal of Computer Information Systems and Industrial Management Applications": [
   "Int. J. Comput. Inf. Syst. Ind. Manage. Appl."],
 "IEEE Geoscience and Remote Sensing Magazine": [
   "IEEE Geosci. Remote Sens. Mag.", "IEEE Geoscience and Remote Sensing Mag."],
 "International Journal of Intelligent Engineering and Systems": [
   "Int. J. Intell. Eng. Syst."],
 "Earth-Science Reviews": ["Earth-Sci. Rev."],
 "Proceedings of the IEEE": ["Proc. IEEE"],
 "Machine Learning": ["Mach. Learn."],
 "Neural Computation": ["Neural Comput."],
 "Neural Networks": ["Neural Netw."],
 "Remote Sensing": ["Remote Sens."],
}

# longest abbreviation first so no shorter key eats part of a longer name
PAIRS = []
for full, abbrs in FULL.items():
    for a in abbrs:
        PAIRS.append((a, full))
PAIRS.sort(key=lambda t: -len(t[0]))

s = io.open(P, encoding="utf-8").read()
head, bib = s.split("\\begin{thebibliography}", 1)

parts = bib.split("\\bibitem")
out = [parts[0]]
n = 0
for blk in parts[1:]:
    flat = re.sub(r"\s+", " ", blk).strip()
    for a, full in PAIRS:
        if a in flat:
            flat = flat.replace(a, full)
            n += 1
    out.append("\n" + flat + "\n")
bib = "\\bibitem".join(out)

s = head + "\\begin{thebibliography}" + bib
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("expanded", n, "venue names across", len(parts) - 1, "entries")

left = [a for a, _ in PAIRS if a in bib]
print("remaining abbreviations:", left if left else "none")
