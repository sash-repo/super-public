#!/usr/bin/env python3
"""Cross-validation of sweep3.py against Theorem 5.1 of the text: run the same engine on
osp_a(4|2) = D(2,1;a) (model of d21a.py) and check that it reproduces
   g_ev = sl(2)+sl(2)+sl(2),  sl(2)+osp(1|2)  (all a),  osp(3|2)  (a=1,-2,-1/2)
and nothing else semisimple.
"""
import sys
from fractions import Fraction as F
from itertools import product
import d21a
from sweep3 import run_grading


class D21a:
    """wrapper giving d21a.py's model the interface sweep3.py expects."""

    def __init__(self, a):
        self.a = a
        self.N = d21a.N
        self.PAR = d21a.PAR
        self.B = d21a.build((F(1), F(a), F(-1) - F(a)))
        assert d21a.check_jacobi(self.B) == 0
        self.name = "osp_a(4|2), a=%s" % a
        self.beta = a
        WT = [None] * self.N
        for m in range(3):
            for kind, w in (("E", 2), ("H", 0), ("F", -2)):
                v = [F(0)] * 3
                v[m] = F(w)
                WT[d21a.sl2(m, kind)] = tuple(v)
        for I in product((0, 1), repeat=3):
            WT[d21a.xi(*I)] = tuple(F(-1) ** i for i in I)
        self.WT = WT

    def br(self, x, y):
        return d21a.bracket(self.B, x, y)

    def e(self, i):
        return [F(1) if k == i else F(0) for k in range(self.N)]


if __name__ == "__main__":
    for a in [F(5), F(1), F(1, 3)]:
        g = D21a(a)
        print("\n" + "#" * 100)
        print("# %s" % g.name)
        run_grading(g, (0, F(1, 2), F(1, 2)), "osp_a(4|2) depth 1 [a=%s]" % a,
                    Vmode="full", dump="runs/D21a-d1-%s.pkl" % a.numerator
                    if a.denominator == 1 else "runs/D21a-d1-%s_%s.pkl"
                    % (a.numerator, a.denominator))
        run_grading(g, (0, 0, 1), "osp_a(4|2) contact [a=%s]" % a,
                    Vmode="full", dump="runs/D21a-c-%s.pkl" % a.numerator
                    if a.denominator == 1 else "runs/D21a-c-%s_%s.pkl"
                    % (a.numerator, a.denominator))
