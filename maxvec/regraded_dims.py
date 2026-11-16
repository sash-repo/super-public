#!/usr/bin/env python3
"""Dimensions of the components of degree 1 and 2 of the regraded algebras
k(1|2n; n) and m(n; n), by counting generating functions.

The degrees are those of formula (53) of the text, divided by their common factor.
k(1|2n; n): generating functions f in t (even), xi_1..xi_n, eta_1..eta_n (odd);
            deg t = deg xi_j = 2, deg eta_j = 0 (all degrees even, so K_f has
            degree deg f / 2 - 1).  The count is symmetric in xi and eta.
m(n; n):    generating functions f in q_1..q_n (even), xi_1..xi_n, tau (odd);
            deg q_j = deg tau = 1, deg xi_j = 0; M_f has degree deg f - 1
            (below, all degrees are doubled, which changes nothing).

These are the numbers against which the Cartan prolongs computed by prol2.py are
identified in Statement 7.5.2: 16, 16 for k(1|4;2) and 20, 28 for m(2;2) in degrees
1, 2; and 56 for k(1|6;3), 72 for m(3;3) in degree 1.
"""
from math import comb


def k_dim(n, i):
    """dim of the degree-i component of k(1|2n; n): f = t^a xi^T eta^S, a + |T| = i + 1."""
    d = i + 1
    return sum(comb(n, T) for a in range(d + 1) for T in [d - a] if T <= n) * 2 ** n


def m_dim(n, i):
    """dim of the degree-i component of m(n; n): f = tau^a q^b xi^S, a + |b| = i + 1, a <= 1."""
    d = i + 1
    total = 0
    for a in (0, 1):
        b = d - a
        if b < 0:
            continue
        total += comb(n + b - 1, b)          # monomials of degree b in n even q's
    return total * 2 ** n


if __name__ == "__main__":
    for n in (2, 3):
        print(f"k(1|{2*n};{n}): dims of degrees 1, 2 = {k_dim(n, 1)}, {k_dim(n, 2)};   "
              f"m({n};{n}): dims of degrees 1, 2 = {m_dim(n, 1)}, {m_dim(n, 2)}")
