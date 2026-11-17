#!/usr/bin/env python3
"""Statement 2.8.3: the operator L_n of formula (70) is anti-self-adjoint,

    Res( f . L_n(g) ) = -(-1)^{n p(f)} Res( L_n(f) . g ),                  (75)

for n <= 3 in the Neveu--Schwarz series, L_n = D_1^{3-n} D_{theta_1} ... D_{theta_n},
D_1 = d_x, D_{theta_i} = theta_i d_x + d_{theta_i}, Res = the coefficient of
x^{-1} theta_1 ... theta_n.  (The Ramond case n = 1 is checked by selfadj.py M.)

The identity is tested on all pairs of monomials x^a theta_S with |a| <= 5.
Exact rational arithmetic; the Grassmann algebra is the one of kdops.py.
Run:  python3 selfadjeq.py
"""
from fractions import Fraction as Q
from itertools import combinations
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from kdops import R


def basis(A, odds, amin=-5, amax=5):
    out = []
    for r in range(len(odds) + 1):
        for S in combinations(odds, r):
            for a in range(amin, amax + 1):
                out.append(A.mono(list(S), a))
    return out


def run(name, A, L, odds, n, top):
    def res(h):
        return h.get((tuple(sorted(A.ix[g] for g in top)), Q(-1)), Q(0))
    B = basis(A, odds)
    tested = fails = nonzero = 0
    for f in B:
        pf = A.parity(f)
        for g in B:
            lhs = res(A.mul(f, L(g)))
            rhs = -((-1) ** (n * pf)) * res(A.mul(L(f), g))
            tested += 1
            nonzero += bool(lhs)
            fails += (lhs != rhs)
    print(f"{name}: {tested} pairs, {nonzero} non-zero values, {fails} failures")
    return fails


def ns(n):
    ths = ["t%d" % i for i in range(1, n + 1)]
    A = R(ths)

    def L(f):
        for g in reversed(ths):                     # D_{theta_n} first
            f = A.add(A.mul(A.gen(g), A.d_x(f)), A.d_odd(g, f))
        for _ in range(3 - n):
            f = A.d_x(f)
        return f
    return run(f"k^L(1|{n}), L_{n} = D_1^{3 - n} D_th1..D_th{n}", A, L, ths, n, ths)


if __name__ == "__main__":
    total = sum(ns(n) for n in (0, 1, 2, 3))
    print("TOTAL failures:", total)
