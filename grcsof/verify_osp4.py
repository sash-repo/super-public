#!/usr/bin/env python3
"""Full verification of the closed formula for the order-4 i(osp(6|2))-invariant
GRC bracket over k(1|6): build the vector from the formula and check it against
ALL the generating functions of i(osp(6|2)), at several pairs of weights."""
from fractions import Fraction as Fr
import sys
from kasbin import build, testmons, ALGS
from kasops import binary_terms, blabel

XI, ETA = [0, 2, 4], [1, 3, 5]
ts = binary_terms(4, True)


def formula(l1, l2):
    v = [Fr(0)] * len(ts)
    def put(t, c):
        v[ts.index(t)] += c
    for i in range(3):
        put((0, (XI[i], ETA[i]), 0, (XI[i], ETA[i])), Fr(1))
        put((1, (), 0, (XI[i], ETA[i])), Fr(-1))
        put((0, (XI[i], ETA[i]), 1, ()), Fr(-1))
        # Omega o (dt (x) 1)  and  Omega o (1 (x) dt)
        put((1, (XI[i],), 0, (ETA[i],)), l2 + 1)
        put((1, (ETA[i],), 0, (XI[i],)), l2 + 1)
        put((0, (XI[i],), 1, (ETA[i],)), -(l1 + 1))
        put((0, (ETA[i],), 1, (XI[i],)), -(l1 + 1))
    for i in range(3):
        for j in range(i + 1, 3):
            for A, B in [((XI[i], XI[j]), (ETA[i], ETA[j])),
                         ((XI[i], ETA[j]), (ETA[i], XI[j])),
                         ((ETA[i], XI[j]), (XI[i], ETA[j])),
                         ((ETA[i], ETA[j]), (XI[i], XI[j]))]:
                put((0, tuple(sorted(A)), 0, tuple(sorted(B))),
                    Fr(-1) * sgn(A) * sgn(B))
    put((2, (), 0, ()), l2 * (l2 + 1) / 2)
    put((0, (), 2, ()), l1 * (l1 + 1) / 2)
    put((1, (), 1, ()), 2 - l1 - l2 - l1 * l2)
    return v


def sgn(A):
    """sign of reordering the pair A increasingly (the D's anticommute up to
    2 B_{ij} dt, and B vanishes on the pairs used here)."""
    return 1 if A[0] < A[1] else -1


if __name__ == '__main__':
    F = ALGS['osp']()
    names = sorted(F)
    for l1, l2 in [(Fr(1), Fr(2)), (Fr(1, 2), Fr(1, 3))]:
        v = formula(l1, l2)
        bad = build([F[k] for k in names], 4, l1, l2, testmons(2, 6),
                    check=[v], names=names)
        print(f"lam = ({l1},{l2}): ",
              "invariant, verified against all generators of i(osp(6|2))"
              if not bad else f"FAILS on {sorted(set(bad))}", flush=True)
