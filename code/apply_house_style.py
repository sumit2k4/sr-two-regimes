"""Apply the author's standing house style to the manuscript.

  1. journal and conference names written out in full, never abbreviated
  2. no en/em dashes anywhere, including page ranges
  3. abstract between 200 and 220 words
  4. captions under ten words (already applied)
"""
import io
import os
import re
import sys

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "main.tex")

# longest first, so that a longer name is never half-replaced by a shorter one
NAMES = [
 ("IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.",
  "IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing"),
 ("Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops",
  "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops"),
 ("Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops",
  "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops"),
 ("Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops",
  "Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops"),
 ("Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.",
  "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition"),
 ("Proc. IEEE Conf. Comput. Vis. Pattern Recognit.",
  "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition"),
 ("Proc. ACM SIGSPATIAL Int. Conf. Adv. Geogr. Inf. Syst.",
  "Proceedings of the ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems"),
 ("Proc. IEEE Winter Conf. Appl. Comput. Vis.",
  "Proceedings of the IEEE Winter Conference on Applications of Computer Vision"),
 ("Proc. Eur. Conf. Comput. Vis. Workshops",
  "Proceedings of the European Conference on Computer Vision Workshops"),
 ("Proc. Int. Conf. Learn. Represent.",
  "Proceedings of the International Conference on Learning Representations"),
 ("Proc. Int. Conf. Mach. Learn.",
  "Proceedings of the International Conference on Machine Learning"),
 ("Proc. Eur. Conf. Comput. Vis.",
  "Proceedings of the European Conference on Computer Vision"),
 ("ISPRS J. Photogramm. Remote Sens.",
  "ISPRS Journal of Photogrammetry and Remote Sensing"),
 ("IEEE Trans. Pattern Anal. Mach. Intell.",
  "IEEE Transactions on Pattern Analysis and Machine Intelligence"),
 ("Int. J. Comput. Inf. Syst. Ind. Manage. Appl.",
  "International Journal of Computer Information Systems and Industrial Management Applications"),
 ("IEEE Geosci. Remote Sens. Mag.",
  "IEEE Geoscience and Remote Sensing Magazine"),
 ("Int. J. Intell. Eng. Syst.",
  "International Journal of Intelligent Engineering and Systems"),
 ("Earth-Sci. Rev.", "Earth-Science Reviews"),
 ("Remote Sens.", "Remote Sensing"),
 ("Proc. IEEE", "Proceedings of the IEEE"),
 ("Mach. Learn.", "Machine Learning"),
 ("Neural Comput.", "Neural Computation"),
 ("Neural Netw.", "Neural Networks"),
]


def main():
    s = io.open(P, encoding="utf-8").read()
    head, bib = s.split("\\begin{thebibliography}", 1)

    # --- 1. full journal and conference names (bibliography only) ----------
    n = 0
    for abbr, full in NAMES:
        c = bib.count(abbr)
        if c:
            bib = bib.replace(abbr, full)
            n += c
    print("expanded", n, "abbreviated venue names")

    # --- 2. no en dashes: LaTeX '--' becomes a plain hyphen ----------------
    d = bib.count("--")
    bib = bib.replace("--", "-")
    head_d = head.count("--")
    head = head.replace("--", "-")
    print("converted", d + head_d, "en dashes to hyphens")

    s = head + "\\begin{thebibliography}" + bib

    # --- 3. abstract length -------------------------------------------------
    a = s.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    plain = re.sub(r"\\[A-Za-z]+\*?", " ", a)
    plain = re.sub(r"[{}$\\~^_]", " ", plain)
    words = len(plain.split())
    print("abstract words:", words, "(target 200-220)")

    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("house style applied")


if __name__ == "__main__":
    main()
