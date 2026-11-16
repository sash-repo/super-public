"""H^2(svect^L_lambda(1|2); svect^L_lambda(1|2)) in the weights w != 0, both
cochain parities, in a window of N steps with core M = 2 (Statement 3.5, item 2)).

Usage:  ../venv/bin/python svect_wide.py [N] [lambdas] [weights]
        e.g.  svect_wide.py 5 1/3,2/5 -4,-3,3,4
"""
import sys, time
from fractions import Fraction as Q
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from defco import Graded, h2_core
import yb


def say(s):
    print(s, flush=True)


N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LAMS = [Q(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [Q(1, 3), Q(2, 5)]
WS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [-4, -3, 3, 4]
for lam in LAMS:
    g = Graded(yb.SVect(lam), "svect^L_{%s}(1|2)" % lam)
    for w in WS:
        for par in (0, 1):
            t0 = time.time()
            r = h2_core(g, N, 2, weight=w, cochain_parity=par)
            say("%s N=%d w=%d par=%d %s  (%.0fs)" % (g.name, N, w, par, r, time.time() - t0))
