#!/usr/bin/env python3
"""Over LGr_n, 3<=n<=7: a retract in Igonin's range whose sub half carries
H^1(X; Omega^1) = C (the Kahler class).  Full table of H^0,H^1,H^2 of both
halves of every even layer T_p.
Items 1) and 2) of Theorem 4.5.4 of the text."""
import sys
from retract import named, levi_weyl, retract, obstruction, dims_of, decompose, weights
from sub2 import sub_coh
from cot import n_plus

CASES = {3: (-2, [1]), 4: (-2, [1, 2]), 5: (-2, [5]), 6: (-2, [3, 4]), 7: (-2, [9])}


def omega1_hw(X, W):
    m = {}
    for b in n_plus(X):
        mu = tuple(-x for x in b)
        m[mu] = m.get(mu, 0) + 1
    return list(decompose(X, m, W).items())


for n in sorted(CASES):
    k, chars = CASES[n]
    X = named("LGr_%d" % n); W = levi_weyl(X); r = X.R.rank
    idw = [0] * r; idw[0] = 1
    parts = [(idw, k)] + [([0] * r, c) for c in chars]
    E = retract(X, parts)
    rank = sum(E.values())
    print("LGr_%d  E = id(x)d^%d + %s, rank %d (range <= %d); Omega^1 = %s"
          % (n, k, " + ".join("d^%d" % c for c in chars), rank, 2 * n - 1,
             omega1_hw(X, W)))
    for p in range(2, rank + 1, 2):
        q = dims_of(obstruction(X, E, p, W)); s = sub_coh(X, E, p, W)
        sd = dims_of(s)
        print("   p=%d quot H0,1,2=%s  sub H0,1,2=%s  sub H^1=%s"
              % (p, [q.get(i, 0) for i in (0, 1, 2)],
                 [sd.get(i, 0) for i in (0, 1, 2)], s.get(1, [])))
    sys.stdout.flush()
