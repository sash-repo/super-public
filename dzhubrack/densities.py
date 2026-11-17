#!/usr/bin/env python3
r"""Footnote 1 and Subsec. 1.5 of arXiv:1906.01380: Dzhumadildaev's question in the
module of weighted densities.

vect(1) acts on F_lambda = {f (dt)^lambda} by

    rho(u d/dt) = u d/dt + lambda u',

so rho(vect(1)) is a Lie subalgebra of the Weyl algebra K[t, d/dt].  We ask for
which k the antisymmetrizor a_k of k elements of rho(vect(1)) again lies in
rho(vect(1)), i.e., which k are critical for the pair (vect(1), F_lambda).

A differential operator is a dict {(a, b): c} standing for c t^a (d/dt)^b; the
arithmetic is exact over Q.  a_k is alternating, so it is enough to run over the
k-subsets of the basis {t^m d/dt} with m <= M.
"""

import sys
from fractions import Fraction
from itertools import combinations
from math import comb

from sant import amult


def falling(c, j):
    r = 1
    for t in range(j):
        r *= (c - t)
    return r


def dmul(u, v):
    out = {}
    for (a, b), c1 in u.items():
        for (cc, d), c2 in v.items():
            for j in range(min(b, cc) + 1):
                key = (a + cc - j, b + d - j)
                out[key] = out.get(key, 0) + c1 * c2 * comb(b, j) * falling(cc, j)
    return {k: v for k, v in out.items() if v != 0}


def dadd(u, v):
    out = dict(u)
    for k, c in v.items():
        out[k] = out.get(k, 0) + c
        if out[k] == 0:
            del out[k]
    return out


def rho(m, lam):
    """rho(t^m d/dt) acting on lambda-densities."""
    d = {(m, 1): Fraction(1)}
    if m > 0 and lam != 0:
        d[(m - 1, 0)] = Fraction(lam) * m
    return d


def as_vector(op, amax, bmax):
    return [op.get((a, b), 0) for a in range(amax + 1) for b in range(bmax + 1)]


def in_rho_span(op, lam, amax, bmax):
    from sant import in_span
    basis = [as_vector(rho(m, lam), amax, bmax) for m in range(amax + 1)]
    return in_span(as_vector(op, amax, bmax), basis)


def survey(lam, ks, M=7):
    res = {}
    for k in ks:
        nonzero, inside, wit = False, True, None
        for S in combinations(range(M + 1), k):
            ops = [rho(m, lam) for m in S]
            A = amult(ops, [0] * k, dmul, dadd, {})
            if A:
                nonzero = True
                amax = max(a for a, b in A) + 1
                bmax = max(b for a, b in A) + 1
                if not in_rho_span(A, lam, amax, bmax):
                    inside, wit = False, S
                    break
        res[k] = (nonzero, inside)
        v = "CRITICAL" if (nonzero and inside) else \
            ("a_k = 0 identically" if not nonzero else f"leaves the algebra (e.g. {wit})")
        print(f"  lambda = {str(lam):>6s}   a_{k:<2d} {v}")
        sys.stdout.flush()
    return res


if __name__ == "__main__":
    lams = [Fraction(0), Fraction(-1), Fraction(1), Fraction(1, 2), Fraction(-1, 2),
            Fraction(2), Fraction(-2), Fraction(3, 7)]
    for lam in lams:
        survey(lam, range(2, 7))
        print()
