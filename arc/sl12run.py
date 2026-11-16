"""Sweep of dim Hom_{sl(1|2)}(F_a (x) F_b, F_c) on S_c^{1|2}.

Writes sl12run.log in this directory (never /tmp).
"""

import sys
import sympy as sp
from sl12 import hom_dim, ansatz_terms

LOG = open("sl12run.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def sweep(wa, wb, Nrange, dmax=6, label=""):
    say("\n=== weights a=%s b=%s   (dmax=%d) %s" % (wa, wb, dmax, label))
    say("  N    q   #terms  dim Hom   target (c1,c2)")
    table = {}
    for N in Nrange:
        for q in range(-3, 4):
            if (q - N) % 2:
                continue
            terms = ansatz_terms(N, q)
            if not terms:
                continue
            d = hom_dim(N, q, wa, wb, dmax=dmax)
            table[(N, q)] = d
            c = (wa[0] + wb[0] + N + q, wa[1] + wb[1] - q)
            say("%3d %4d %7d %8d      %s" % (N, q, len(terms), d, c))
    return table


if __name__ == "__main__":
    a1, a2, b1, b2 = sp.symbols('a1 a2 b1 b2')
    which = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if which == "gen":
        sweep((a1, a2), (b1, b2), range(-2, 5), dmax=6, label="generic")
    else:
        sweep((sp.Integer(1), sp.Integer(0)), (sp.Integer(1), sp.Integer(0)),
              range(-2, 5), dmax=6)
    LOG.close()
