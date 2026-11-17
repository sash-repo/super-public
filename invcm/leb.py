#!/usr/bin/env python3
"""Statement 2.7.5: formula (22) for B_kj against the recursion of [Leb] = arXiv:2311.02710.

[Leb], Statement 2.1: with d_{-1} = 0 and d_m = (-1)^{i_k}(d_{m-1} - A_kj - m A_kk),
one has B_kj = inf{ m >= 0 | d_m = 0 }, the length of the alpha_k-string through alpha_j.

Formula (22) of the text (formula (3) of [Leb] in our notation) is the closed expression
for the same B_kj in terms of A_kk, A_kj and the parity i_k.  This script checks that the two
agree for every pair (A_kk, A_kj) of elements of F_p and of F_{p^2}, p = 2, 3, 5, 7, and both
parities -- F_{p^2} is what exercises the two cases of (22) where A_kj/A_kk is not in Z/pZ --
and counts the triples where the formula that (22) replaces (formula (7.4) of [BGL],
described in Remark 2.7.4) disagrees with the recursion.

Exact arithmetic throughout (integers mod p; F_{p^2} as pairs modulo an irreducible
quadratic).

Usage:  python leb.py
"""

import itertools
import sys


class GF2:
    """F_{p^2} = F_p[t]/(t^2 - c t - d), elements are pairs (a, b) = a + b t."""

    def __init__(self, p, c, d):
        self.p, self.c, self.d = p, c, d

    def elements(self):
        return [(a, b) for a in range(self.p) for b in range(self.p)]

    def add(self, x, y):
        return ((x[0] + y[0]) % self.p, (x[1] + y[1]) % self.p)

    def sub(self, x, y):
        return ((x[0] - y[0]) % self.p, (x[1] - y[1]) % self.p)

    def neg(self, x):
        return ((-x[0]) % self.p, (-x[1]) % self.p)

    def mul(self, x, y):
        # (a + b t)(e + f t) = ae + (af + be) t + bf t^2, t^2 = c t + d
        a, b = x
        e, f = y
        p, c, d = self.p, self.c, self.d
        return ((a * e + b * f * d) % p, (a * f + b * e + b * f * c) % p)

    def smul(self, n, x):
        return ((n * x[0]) % self.p, (n * x[1]) % self.p)

    def inv(self, x):
        for y in self.elements():
            if self.mul(x, y) == (1, 0):
                return y
        raise ZeroDivisionError(x)

    def is_zero(self, x):
        return x == (0, 0)

    def in_prime_field(self, x):
        return x[1] == 0

    def to_int(self, x):
        assert x[1] == 0
        return x[0]

    def from_int(self, n):
        return (n % self.p, 0)


def irreducible(p):
    """Return (c, d) with x^2 - c x - d irreducible over F_p."""
    for c in range(p):
        for d in range(p):
            # roots of x^2 - c x - d = 0 ?
            if all((r * r - c * r - d) % p != 0 for r in range(p)):
                return c, d
    raise RuntimeError(p)


def recursion(K, ik, Akk, Akj, bound):
    """First m >= 0 with d_m = 0, or None if none is found below `bound`."""
    d = (0, 0)
    sign = 1 if ik == 0 else -1
    for m in range(bound):
        # d_m = (-1)^{i_k} (d_{m-1} - A_kj - m A_kk)
        t = K.sub(K.sub(d, Akj), K.smul(m, Akk))
        d = t if sign == 1 else K.neg(t)
        if K.is_zero(d):
            return m
    return None


def closed(K, ik, Akk, Akj):
    """Formula (22) of the text = [Leb] formula (3)."""
    p = K.p
    if K.is_zero(Akj):
        return 0
    if ik == 0:
        if K.is_zero(Akk):
            return p - 1
        if p == 2:
            return 2 if Akj == Akk else 3
        ratio = K.mul(Akj, K.inv(Akk))
        if K.in_prime_field(ratio):
            #  ( -2 A_kj / A_kk )_Z
            return (-2 * K.to_int(ratio)) % p
        return p - 1
    else:
        if K.is_zero(Akk):
            return 1
        ratio = K.mul(Akj, K.inv(Akk))
        if K.in_prime_field(ratio):
            #  2 ( -A_kj / A_kk )_Z
            return 2 * ((-K.to_int(ratio)) % p)
        return 2 * p - 1


def bgl(K, ik, Akk, Akj):
    """The formula that (22) replaces, [BGL] (7.4): the odd non-isotropic node is given
    the even node's values, and p = 2 needs a separate convention (Remark 2.7.4)."""
    p = K.p
    if K.is_zero(Akk) and K.is_zero(Akj):
        return 0
    if K.is_zero(Akk):           # isotropic node, A_kj != 0
        return 1 if ik == 1 else p - 1
    ratio = K.mul(Akj, K.inv(Akk))
    if K.in_prime_field(ratio):
        return (-2 * K.to_int(ratio)) % p
    return p - 1


def main():
    def out(s=""):
        print(s)

    out("Check of formula (22) = [Leb] (3) against the recursion of [Leb], Statement 2.1,")
    out("exhaustively over F_p and F_{p^2}.")
    out("")
    total = bad = differ = 0
    for p in (2, 3, 5, 7):
        c, d = irreducible(p)
        K = GF2(p, c, d)
        n_pf = n_ext = 0
        diff_here = []
        for ik in (0, 1):
            for Akk, Akj in itertools.product(K.elements(), repeat=2):
                bound = 4 * p + 4
                m = recursion(K, ik, Akk, Akj, bound)
                f = closed(K, ik, Akk, Akj)
                total += 1
                if m != f:
                    bad += 1
                    out("  MISMATCH p=%d i_k=%d A_kk=%s A_kj=%s: "
                        "recursion %s, formula (22) %s" % (p, ik, Akk, Akj, m, f))
                if K.in_prime_field(Akk) and K.in_prime_field(Akj):
                    n_pf += 1
                else:
                    n_ext += 1
                g = bgl(K, ik, Akk, Akj)
                if g != f:
                    differ += 1
                    diff_here.append((ik, Akk, Akj, g, f))
        out("p = %d: %d pairs over F_p, %d more over F_{p^2}; "
            "x^2 - %dx - %d irreducible" % (p, n_pf, n_ext, c, d))
        out("        formula (22) differs from [BGL] (7.4) in %d of them; "
            "e.g." % len(diff_here))
        for ik, Akk, Akj, g, f in diff_here[:4]:
            out("        i_k=%s A_kk=%s A_kj=%s: (7.4) gives %d, (22) gives %d"
                % ("ev" if ik == 0 else "od", Akk, Akj, g, f))
    out("")
    out("checked %d triples (i_k, A_kk, A_kj); mismatches with the recursion: %d"
        % (total, bad))
    out("triples where [BGL] (7.4) disagrees with (22): %d" % differ)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
