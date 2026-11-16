#!/usr/bin/env python3
"""k(1|5): the shape of the odd bracket of order 5.

The order-5 space is 1-dimensional at generic weights (n5bin.py).  Its solution
contains the two extreme terms T_5 (x) 1 and 1 (x) T_5, where

    T_5 = D_th (D_xi1 D_eta1 - d_t)(D_xi2 D_eta2 - d_t)

is the Bol operator of n5unary.py.  Normalizing the coefficient of T_5 (x) 1 to
1, this script reads off the coefficient of 1 (x) T_5 at several weight pairs
and compares it with the conjecture

    coeff(1 (x) T_5) = - prod_{0<=i<=4}(lam1+i) / prod_{0<=i<=4}(lam2+i),

which is the n = 5 instance of the pattern seen for n = 3, where the analogous
ratio is prod_{0<=i<=2}(lam1+i) / prod_{0<=i<=2}(lam2+i).
"""
import sys
from fractions import Fraction as Fr

from n5bin import Bn, gens_osp
from n5 import ALG

NM = ALG.names

# the four terms of T_5 in the ansatz basis (kA, A, kB, B), left and right slot
XI1, ETA1, XI2, ETA2, TH = 1, 2, 3, 4, 0
T5_TERMS = [((2, (TH,)), 1), ((1, (TH, XI1, ETA1)), -1),
            ((1, (TH, XI2, ETA2)), -1), ((0, (TH, XI1, ETA1, XI2, ETA2)), 1)]


def conj(l1, l2):
    num = 1
    den = 1
    for i in range(5):
        num *= (l1 + i)
        den *= (l2 + i)
    return -num / den


def coeffs_of(ts, v):
    d = {t: c for t, c in zip(ts, v) if c != 0}
    left = {}
    right = {}
    for (k, A), s in T5_TERMS:
        left[(k, A)] = d.get((k, A, 0, ()), 0) / s
        right[(k, A)] = d.get((0, (), k, A), 0) / s
    return left, right


PAIRS = [(Fr(3, 7), Fr(-5, 11)), (Fr(1, 3), Fr(2, 5)), (Fr(-7, 3), Fr(5, 2)),
         (Fr(2), Fr(-3, 4)), (Fr(1, 2), Fr(1, 2))]

if __name__ == '__main__':
    for l1, l2 in PAIRS:
        C = Bn(gens_osp(), l1, l2)
        ts, ns = C.solve(5, tmax=2)
        if len(ns) != 1:
            print(f"lam1={l1}, lam2={l2}: dim = {len(ns)} -- skipped")
            sys.stdout.flush()
            continue
        left, right = coeffs_of(ts, ns[0])
        ls = set(left.values())
        rs = set(right.values())
        ok_shape = len(ls) == 1 and len(rs) == 1
        a, b = left[(2, (TH,))], right[(2, (TH,))]
        r = b / a if a else None
        print(f"lam1={l1}, lam2={l2}: T_5(x)1 and 1(x)T_5 present as whole "
              f"blocks: {ok_shape};  ratio = {r},  conjecture = {conj(l1, l2)}"
              f"  -> {'MATCH' if r == conj(l1, l2) else 'DIFFERS'}")
        sys.stdout.flush()
