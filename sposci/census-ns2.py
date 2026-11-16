#!/usr/bin/env python3
r"""census-ns2.py -- (c, h, H) of Spin(V (+) V^#) over k^L(1|2) and k^M(1|2)
on a wide grid of parameters, at two truncation windows."""

from fractions import Fraction as Q
import sys

from ns2 import mono, XI, ETA
import runns2 as R

WINS = (4, 5)
N = 6


def run():
    gridL = [(Q(l, 3), Q(n, 4), Q(m, 5))
             for l in (-2, 0, 1, 3, 7) for n in (-3, 0, 2, 5) for m in (0, 1, 3)]
    gridM = [(Q(l, 3), Q(m, 5)) for l in (-2, 0, 1, 3, 7, 11)
             for m in (0, 1, 2, 3, 4)]
    print("k^L(1|2):  %d triples (lambda, nu, mu)" % len(gridL))
    seen = {}
    for par in gridL:
        W = R.build_W("L", par, N, R.DUAL_L)
        if W is None:
            print("   %s: NO invariant pairing" % (par,))
            continue
        vals = [(3 * R.alpha_from_LL(W, R.br_L, w), R.h_of(W, w), R.H_of(W, w))
                for w in WINS]
        assert vals[0] == vals[1], ("unstable", par, vals)
        seen.setdefault(vals[0], []).append(par)
    for v, ps in seen.items():
        print("   (c, h, H) = (%s, %s, %s) on %d triples" % (v[0], v[1], v[2], len(ps)))
    print("k^M(1|2):  %d pairs (lambda, mu)" % len(gridM))
    seenM = {}
    for par in gridM:
        W = R.build_W("M", par, N, R.DUAL_M)
        if W is None:
            print("   %s: NO invariant pairing" % (par,))
            continue
        vals = [(3 * R.alpha_from_LL(W, R.brM, w), R.h_of(W, w)) for w in WINS]
        assert vals[0] == vals[1], ("unstable", par, vals)
        seenM.setdefault(vals[0], []).append(par)
    for v, ps in seenM.items():
        print("   (c, h) = (%s, %s) on %d pairs" % (v[0], v[1], len(ps)))


if __name__ == "__main__":
    run()
