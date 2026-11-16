#!/usr/bin/env python3
r"""sixs.py --- the six odd elements xi, eta, p xi, q xi, p eta, q eta of Bj(3;1|3)
(each with (ad_s)^2 = 0) and the subalgebras they generate, over F_3:

  * all six generate a 13-dimensional subalgebra equal to its own derived algebra;
  * the five xi, eta, q xi, p eta, q eta generate a 10-dimensional subalgebra whose
    lower central series stabilizes at 7.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runbj import lie_closure, lower_central                          # noqa: E402
from runbj6 import SIX                                                # noqa: E402


def main():
    six = dict(SIX)
    for names in (["xi", "eta", "p xi", "q xi", "p eta", "q eta"],
                  ["xi", "eta", "q xi", "p eta", "q eta"]):
        rows, piv = lie_closure([six[n] for n in names])
        lcs = lower_central(rows, piv)
        print("%s: generate dim %d, lower central series %s%s"
              % (", ".join(names), len(rows), lcs,
                 "  -- PERFECT" if lcs[1] == lcs[0] else ""))


if __name__ == "__main__":
    main()
