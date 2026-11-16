#!/usr/bin/env python3
r"""census-ns4.py -- the free-field count of the central charge.

W = V (+) V^# splits into 2^n commuting systems, one per pair (S, S^c) of
complementary subsets of the odd indeterminates: the tower over S in V and the
tower over S^c in V^# are dual to each other, and their conformal weights,
        h_S = (|S| + lambda) / 2      and      1 - h_S,
add up to 1, so each pair is a (b, c), resp. a (beta, gamma), system.  In
Spin(W) the parity is shifted, so a tower over an EVEN S carries fermionic
operators and one over an odd S bosonic ones; a (b, c) system of weight h
contributes -F(h) and a (beta, gamma) system +F(h), where

        F(h) := 2(6h^2 - 6h + 1).

Hence, writing u := lambda/2,
        c = sum_{0<=j<=n} (-1)^{j+1} binom(n, j) F(u + j/2),
which is an alternating binomial sum of a QUADRATIC function of j, i.e., the
n-th finite difference of a quadratic: it vanishes identically for n >= 3.

This script checks that formula against the four values the exact
Clifford-Weyl computations of Section 4.3 produce, n = 0, 1, 2 and 4.
"""

from fractions import Fraction as Q
from math import comb
import sympy as sp

lam = sp.Symbol("lambda")


def F(h):
    return 2 * (6 * h ** 2 - 6 * h + 1)


def c_of(n, lam):
    if n == 0:                       # deg t = 1 there, so the weight is lambda
        return sp.expand(-F(lam))
    u = lam / 2
    return sp.expand(sum((-1) ** (j + 1) * comb(n, j) * F(u + sp.Rational(j, 2))
                         for j in range(n + 1)))


KNOWN = {0: -(12 * lam ** 2 - 12 * lam + 2),      # eq. (ccZero), Subsec. 4.3.7
         1: 6 * lam - 3,                          # eq. equation 181
         2: sp.Integer(-6),                       # eq. equation 187
         3: sp.Integer(0),                        # not computed here
         4: sp.Integer(0)}                        # eq. equation 194

print("n   free-field count            computed in the text        agree")
for n in range(5):
    got = c_of(n, lam)
    want = sp.expand(KNOWN[n])
    print("%d   %-27s %-27s %s"
          % (n, got, want, "yes" if sp.simplify(got - want) == 0 else "NO"))

print()
print("The table of Subsection ssHW prints minus each of these, in every")
print("column: 12l^2-12l+2, -6l+3, 6, 0.")
