#!/usr/bin/env python3
"""k(1|5): independent verification of the formulas printed in the paper.

Every operator below is written out here BY HAND, from the formula as it is
printed, and fed to verify_kn.check -- a routine that shares no code with the
solver: it re-derives the invariance identity from scratch, for ALL 23
generating functions of i(osp(5|2)) and on a large set of monomials.  So a PASS
here says that the printed formula is invariant, not that the solver's output
was copied correctly.

  J2   the contact bracket, order 2
  J4   the order-4 bracket
  J5   the odd bracket of order 5 (T_5 (x) 1 - R.1 (x) T_5 + ...) -- for this
       one only the two extreme blocks and the ratio R are checked by hand;
       the middle is taken from the solver, see n5fit.py.

Usage:  python n5verify.py [NPAIRS]      (default: all five weight pairs)
"""
import sys
from fractions import Fraction as Fr

from n5 import ALG
from n5bin import Bn, gens_osp
from verify_kn import check, osp_fields
from knclass import label

TH, XI1, ETA1, XI2, ETA2 = 0, 1, 2, 3, 4
PAIRS = [(XI1, ETA1), (XI2, ETA2)]


def J2(l1, l2):
    """sum C_ij D_i (x) D_j - lam1 . 1 (x) d_t + lam2 . d_t (x) 1."""
    d = {(0, (), 1, ()): -l1, (1, (), 0, ()): l2,
         (0, (TH,), 0, (TH,)): 1}
    for x, e in PAIRS:
        d[(0, (x,), 0, (e,))] = 1
        d[(0, (e,), 0, (x,))] = 1
    return d


def J4(l1, l2):
    """the order-4 bracket, as printed."""
    d = {(0, (), 2, ()): l1 * (l1 + 1) / 2,
         (1, (), 1, ()): -l1 * l2 - l1 - l2 + 1,
         (2, (), 0, ()): l2 * (l2 + 1) / 2}
    # -(lam1+1) sum C_ij D_i (x) d_t D_j  +  (lam2+1) sum C_ij d_t D_i (x) D_j
    d[(0, (TH,), 1, (TH,))] = -l1 - 1
    d[(1, (TH,), 0, (TH,))] = l2 + 1
    for x, e in PAIRS:
        d[(0, (x,), 1, (e,))] = -l1 - 1
        d[(1, (x,), 0, (e,))] = l2 + 1
        d[(0, (e,), 1, (x,))] = -l1 - 1
        d[(1, (e,), 0, (x,))] = l2 + 1
        # - d_t (x) D_x D_e  -  D_x D_e (x) d_t
        d[(1, (), 0, (x, e))] = -1
        d[(0, (x, e), 1, ())] = -1
        # the second Clifford contraction, diagonal part
        d[(0, (x, e), 0, (x, e))] = 1
    # ... and its off-diagonal part
    d[(0, (TH, XI1), 0, (TH, ETA1))] = -1
    d[(0, (TH, ETA1), 0, (TH, XI1))] = -1
    d[(0, (TH, XI2), 0, (TH, ETA2))] = -1
    d[(0, (TH, ETA2), 0, (TH, XI2))] = -1
    d[(0, (XI1, XI2), 0, (ETA1, ETA2))] = -1
    d[(0, (XI1, ETA2), 0, (ETA1, XI2))] = -1
    d[(0, (ETA1, XI2), 0, (XI1, ETA2))] = -1
    d[(0, (ETA1, ETA2), 0, (XI1, XI2))] = -1
    return d


def ratio5(l1, l2):
    num, den = 1, 1
    for i in range(5):
        num *= (l1 + i)
        den *= (l2 + i)
    return 'indeterminate (0/0)' if den == 0 else -num / den


def run(name, mk, N, pairs, vtmax=3):
    fields = osp_fields(ALG)
    for l1, l2 in pairs:
        d = mk(l1, l2)
        ts = sorted(d)
        v = [d[t] for t in ts]
        print(f"{name}, order {N}, lam1={l1}, lam2={l2}: {len(ts)} terms")
        ok = check(ALG, ts, v, l1, l2, N, fields, tmax=vtmax)
        print("   ->", "PASS" if ok else "FAIL")
        sys.stdout.flush()


if __name__ == '__main__':
    pairs = [(Fr(3, 7), Fr(-5, 11)), (Fr(1, 3), Fr(2, 5)), (Fr(-7, 3), Fr(5, 2)),
             (Fr(0), Fr(0)), (Fr(-4), Fr(1, 2))]
    if len(sys.argv) > 1:              # optional: only the first NPAIRS pairs
        pairs = pairs[:int(sys.argv[1])]
    run("J2 (the contact bracket)", J2, 2, pairs)
    run("J4", J4, 4, pairs)
    # the order-5 ratio, at the same weight pairs
    print("--- order 5: the ratio coeff(1 (x) T_5) / coeff(T_5 (x) 1)")
    for l1, l2 in pairs:
        C = Bn(gens_osp(), l1, l2)
        ts, ns = C.solve(5, tmax=2)
        print(f"   lam1={l1}, lam2={l2}: dim = {len(ns)}", end='')
        if len(ns) == 1:
            d = {t: c for t, c in zip(ts, ns[0]) if c != 0}
            a = d.get((2, (TH,), 0, ()), 0)
            b = d.get((0, (), 2, (TH,)), 0)
            r = (b / a) if a else 'inf'
            print(f", ratio = {r}, conjecture = {ratio5(l1, l2)}", end='')
        print()
        sys.stdout.flush()
