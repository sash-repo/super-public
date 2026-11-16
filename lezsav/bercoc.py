#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bercoc.py -- formula (D3.69) of Appendix D3 of Berezin's book, 2nd ed.
(p. 259), against the cocycle of Statement 4.2.8 of our text.

(D3.69) gives the non-zero values of the cocycle of the central extension of
svect^L_lambda(1|2), in the basis (D3.68'):

    L_m = t^m ( t d_t + (lambda+m+1)/2 (th1 d_1 + th2 d_2) ),
    S^j_m = t^m th_j ( t d_t + (lambda+m+1)(th1 d_1 + th2 d_2) ),

as

    c(L_m, L_n)              = 1/2 m (m^2 - (lambda+1)^2) delta_{m,-n},
    c(t^k d_{th_i}, S^j_m)   = -m (m - (lambda+1)) delta_{...} delta_{i,j},
    c(t^m(th1 d_1 - th2 d_2), t^n(th1 d_1 - th2 d_2)) = m delta_{m,-n},
    c(t^m th1 d_2, t^n th2 d_1)                        = m delta_{m,-n}.

In our labels (sv2.py, sv2coc.py):
    the book's L_m is -L_m of ours (the two conventions for witt differ by a
        sign; the value c(L,L) is unaffected, being bilinear in it);
    t^k d_{th_i}  = G_i^{k-1/2};
    S^j_m         = Gbar^{(m+1/2) j}   (verified in berroots.py's calculus);
    t^m th1 d_2   = J^+_m,   t^m th2 d_1 = J^-_m,
    t^m(th1 d_1 - th2 d_2) = J^0_m.

The second line of (D3.69) is garbled in print: the first argument carries the
mode k, the Kronecker delta is written delta_{m,-n} with an n that occurs
nowhere, and the polynomial is written in m.  So there are two readings, and
the third and fourth lines admit the printed value m or the value 2m for
c(J^0, J^0).  All four combinations are tested here; exactly one of them is a
2-cocycle, and it is cohomologous to the cocycle of Statement 4.2.8.

Run: ../venv/bin/python bercoc.py
"""

import os
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from sympy import Rational as R, expand, simplify, solve, symbols

import sv2coc as C

L_, JP, JM, J0, G1, G2, B1, B2 = range(8)
NAMES = C.NAMES
PAR = C.PAR

lam = symbols('lambda')

FAIL = []
CHECKS = [0]


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAIL.append(msg)
        print("  FAIL: " + msg)


# --------------------------------------------------------- the book's cochain

def cbook(u, v, l, odd_sign, j0_factor):
    """(D3.69) in our labels.

    odd_sign = -1 : the polynomial is written in the mode m of S^j_m, as
                    printed,  -m(m - (l+1));
    odd_sign = +1 : it is written in the mode k of the first argument,
                    -k(k - (l+1)), which in the mode of S^j_m reads
                    -m(m + (l+1)).
    j0_factor    : the value of c(J^0_m, J^0_{-m}) is j0_factor * m.
    """
    (i, m), (j, n) = u, v
    if m + n != 0:
        return 0
    if (i, j) == (L_, L_):
        return R(1, 2) * m * (m ** 2 - (l + 1) ** 2)
    if (i, j) == (JP, JM):
        return m
    if (i, j) == (JM, JP):
        return m
    if (i, j) == (J0, J0):
        return j0_factor * m
    # the odd pairs: G_i^r with r = m,  Gbar^{s j} with s = n = -m
    if (i, j) in ((G1, B1), (G2, B2)):
        s = n                       # the mode of Gbar = the S^j_{s-1/2}
        ms = s - R(1, 2)            # the book's m in S^j_m
        return -ms * (ms + odd_sign * (l + 1))
    if (i, j) in ((B1, G1), (B2, G2)):
        s = m
        ms = s - R(1, 2)
        return -ms * (ms + odd_sign * (l + 1))
    return 0


def cours(u, v, l):
    return C.cclosed(u, v, l)


# ------------------------------------------------- the cocycle identity test

def modes(p, M):
    if p == 0:
        return [R(k) for k in range(-M, M + 1)]
    return [R(2 * k + 1, 2) for k in range(-M - 1, M + 1)]


def cocycle_defects(c, l, M=4, limit=6):
    """How often the identity  sum (-1)^{|i||k|} c([u,v],w) = 0  fails."""
    bad, n, shown = 0, 0, []
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for m in modes(PAR[i], M):
                    for p in modes(PAR[j], M):
                        q = -m - p
                        if q not in modes(PAR[k], M):
                            continue
                        tot = 0
                        for (a, b, w, s) in (
                                ((i, m), (j, p), (k, q),
                                 (-1) ** (PAR[i] * PAR[k])),
                                ((j, p), (k, q), (i, m),
                                 (-1) ** (PAR[j] * PAR[i])),
                                ((k, q), (i, m), (j, p),
                                 (-1) ** (PAR[k] * PAR[j]))):
                            for (t, uu), co in C.br(a, b, l).items():
                                tot += s * co * c((t, uu), w, l)
                        tot = expand(tot)
                        n += 1
                        if tot != 0:
                            bad += 1
                            if len(shown) < limit:
                                shown.append("%s_%s, %s_%s, %s_%s -> %s"
                                             % (NAMES[i], m, NAMES[j], p,
                                                NAMES[k], q, tot))
    return bad, n, shown


# ------------------------------------------------------ cohomologous or not

def is_coboundary_difference(c1, c2, l, M=5):
    """Solve  c1 - c2 = d f  for a degree-zero 1-cochain f, i.e. one supported
    on the mode-0 part;  (d f)(u,v) = - f([u,v]).  Returns the solution or
    None, and verifies it on the whole window."""
    fL, fp, fm, f0 = symbols('f_L f_Jp f_Jm f_J0')
    fs = {L_: fL, JP: fp, JM: fm, J0: f0}
    eqs = []
    pairs = []
    for i in range(8):
        for j in range(8):
            for m in modes(PAR[i], M):
                n = -m
                if n not in modes(PAR[j], M):
                    continue
                d = expand(c1((i, m), (j, n), l) - c2((i, m), (j, n), l))
                cob = 0
                for (t, uu), co in C.br((i, m), (j, n), l).items():
                    if uu == 0 and t in fs:
                        cob += -co * fs[t]
                eqs.append(expand(d - cob))
                pairs.append(((i, m), (j, n)))
    sol = solve([e for e in eqs if e != 0], [fL, fp, fm, f0], dict=True)
    if not sol:
        return None, eqs
    s = sol[0]
    # verify
    bad = 0
    for e in eqs:
        if expand(e.subs(s)) != 0:
            bad += 1
    return (s if bad == 0 else None), eqs


# ==================================================================== main

def main():
    print("=" * 78)
    print("Formula (D3.69) of App. D3 of Berezin's book (p. 259) vs our")
    print("Statement 4.2.8, lambda formal")
    print("=" * 78)

    print("\n0. Our own cocycle satisfies the identity (control)")
    bad, n, _ = cocycle_defects(cours, lam, M=3)
    check(bad == 0, "our own closed-form cocycle fails %d times" % bad)
    print("   %d triples, %d failures" % (n, bad))

    print("\n1. The four readings of (D3.69)")
    good = []
    for odd_sign, tag_s in ((-1, "as printed, -m(m-(l+1))"),
                            (+1, "in the mode k, -k(k-(l+1))")):
        for j0 in (1, 2):
            c = lambda u, v, l, o=odd_sign, jf=j0: cbook(u, v, l, o, jf)
            bad, n, shown = cocycle_defects(c, lam, M=3)
            print("   odd line %-28s c(J^0,J^0)=%dm :  %6d triples, "
                  "%4d failures" % (tag_s, j0, n, bad))
            for s in shown[:2]:
                print("        e.g. " + s)
            if bad == 0:
                good.append((odd_sign, j0))
    check(len(good) == 1, "not exactly one reading of (D3.69) is a cocycle: %s"
          % (good,))
    print("   => the unique reading that is a 2-cocycle: odd sign %+d, "
          "c(J^0,J^0) = %dm" % good[0] if good else "   => none")

    if not good:
        return 1
    odd_sign, j0 = good[0]
    c = lambda u, v, l: cbook(u, v, l, odd_sign, j0)

    print("\n2. Is it our cocycle?")
    sol, eqs = is_coboundary_difference(c, cours, lam, M=4)
    check(sol is not None, "(D3.69) is not cohomologous to Statement 4.2.8")
    if sol is not None:
        print("   yes: (D3.69) - (our cocycle) = d f  with")
        for k, v in sol.items():
            print("      %s = %s" % (k, simplify(v)))
        print("   verified on every pair of the window |mode| <= 4 "
              "(%d equations)" % len(eqs))

    print("\n3. The same, at four numerical values of lambda")
    for l in (R(0), R(1), R(1, 3), R(-5, 2)):
        bad, n, _ = cocycle_defects(
            lambda u, v, ll: cbook(u, v, ll, odd_sign, j0), l, M=3)
        check(bad == 0, "the corrected (D3.69) fails at lambda=%s" % l)
        print("   lambda = %-6s : %6d triples, %d failures" % (l, n, bad))

    print("\n" + "=" * 78)
    print("checks: %d, failures: %d" % (CHECKS[0], len(FAIL)))
    for f in FAIL:
        print("   " + f)
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
