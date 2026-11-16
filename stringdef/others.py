"""Are there OTHER deformations of svect^L_lambda(1|2) than the lambda one?

The family parameter lambda is itself a deformation parameter.  The Buttin
superalgebras b_lambda(n) show that there may be others, some of them with an
ODD parameter.  The question has two halves:

  a) even parameters, weight 0 -- answered by `census.py` (section 3):
     h = 1 at lambda_0 = 1/3 and 2/5, stably at N = 3, 4, 5 with M = 2.
     One class: d mu / d lambda and nothing else.

  b) ODD parameters, and the weights w != 0 -- this module.  The weights
     w != 0 matter for this algebra and for no other member of the list, its
     grading element being OUTER (Lemma 2.2).

The window convention is defco's: Z is computed on a window of N steps and
restricted to a core of M steps.  A count is trustworthy only where it is
STABLE in N: at the edge of too small a window the coboundaries are truncated
and cease to be cocycles, which shows up as a negative h.  Hence every line
below is run at N = 3, 4, 5 and only a stable value is quoted.
"""

from fractions import Fraction as Q
import sys, os, time

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

import yb
from defco import Graded, h2_core, algebras

def say(s=""):
    print(s, flush=True)


def line(g, N, M, w, q):
    t0 = time.time()
    r = h2_core(g, N, M, weight=w, cochain_parity=q)
    say("  %-18s w=%+d q=%d N=%-2d M=%-2d  unk=%-6d rows=%-7d "
        "Z|c=%-4d B|c=%-4d  h=%-3d  [%.1fs]"
        % (g.name, w, q, N, M, r.get("unknowns", 0), r.get("rows", 0),
           r.get("Z", 0), r.get("B", 0), r["h"], time.time() - t0))
    return r["h"]


def main():
    M = 2
    say("\n" + "=" * 78)
    say("Other deformations: odd cochains and the weights w != 0   %s"
        % time.strftime("%Y-%m-%d %H:%M"))
    say("=" * 78)

    say("\n--- A. odd cochains (deformations with an ODD parameter), weight 0")
    for g in algebras(laurent=True):
        for N in (3, 4, 5):
            try:
                line(g, N, M, 0, 1)
            except Exception as e:
                say("  %-18s N=%-2d ERROR %s: %s" % (g.name, N, type(e).__name__, e))
                break

    say("\n--- B. the family svect^L_lambda(1|2): odd cochains, weight 0")
    for lam in (Q(1, 3), Q(2, 5)):
        g = Graded(yb.SVect(lam), "svect^L_{%s}(1|2)" % lam)
        for N in (3, 4, 5):
            line(g, N, M, 0, 1)

    say("\n--- C. the family: the weights w != 0, both parities")
    for lam in (Q(1, 3),):
        g = Graded(yb.SVect(lam), "svect^L_{%s}(1|2)" % lam)
        for q in (0, 1):
            for w in (-2, -1, 1, 2):
                for N in (3, 4):
                    line(g, N, M, w, q)

    say("\ndone")


if __name__ == "__main__":
    main()
