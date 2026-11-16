#!/usr/bin/env python3
"""Remark 8.3.1 of the text: Delta = -Delta (with multiplicities and parities) for k^L(1|n), k^M(1|n)?
Basis xi_i, eta_i (i<=n//2), theta if n odd; root of phi t^k: ((2k+deg phi-2), delta-weights).
Ramond: theta (n odd) or, for n even, the pair is broken: one zeta with t^{1/2+Z}."""
import itertools
from fractions import Fraction
from collections import Counter
def roots(n, ramond, K=6):
    r = n // 2
    odd = [("x", i) for i in range(r)] + [("y", i) for i in range(r)]
    if n % 2: odd.append(("z", 0))
    C = Counter()
    for k2 in range(-2 * K, 2 * K + 1):
        for d in range(len(odd) + 1):
            for S in itertools.combinations(odd, d):
                k = Fraction(k2, 2)
                half = ramond and ("z", 0) in S
                if (k.denominator == 2) != half: continue
                w = [0] * r
                for (c, i) in S:
                    if c == "x": w[i] += 1
                    if c == "y": w[i] -= 1
                C[(2 * k + d - 2, tuple(w), d % 2)] += 1
    return C
for n in range(1, 8):
    for ram in (False, True):
        if ram and n % 2 == 0:
            continue   # Ramond with even n: take n odd with a twisted theta (k^M(1|n), theta = zeta)
        C = roots(n, ram); lim = 6
        bad = [a for a in C if abs(a[0]) <= lim and C[a] != C[(-a[0], tuple(-x for x in a[1]), a[2])]]
        print("%s(1|%d): %d roots in |eps|<=%d checked, asymmetric: %d" % ("k^M" if ram else "k^L", n, sum(1 for a in C if abs(a[0]) <= lim), lim, len(bad)))
