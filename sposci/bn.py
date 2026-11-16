#!/usr/bin/env python3
r"""
bn.py -- the Buttin superalgebra b(n) and the cocycle that quantizes it.

b(n) = Pi( C[q_1..q_n, xi_1..xi_n] ) with the antibracket

    {f, g} = sum_i ( df/dq_i dg/dxi_i + (-1)^{p(f)} df/dxi_i dg/dq_i ),

p(f) being the parity of f AS A FUNCTION; the parity in b(n) is p(f)+1, which
is what makes the antibracket a Lie superalgebra bracket.  Standard grading
deg q_i = deg xi_i = 1, so b(n) = + _{i >= -2} b(n)_i with

    b(n)_{-2} = C.1,   b(n)_{-1} = Span(q, xi),   b(n)_0 = quadratic elements,

and ab(n) = b(n)_{-2} + b(n)_{-1}.  The dictionary with abfock.py is
    ab's z <-> 1,   ab's q_j <-> xi_j,   ab's theta_j <-> q_j.

Table 1 of [LSh] gives the cocycle that quantizes the antibracket:

    C(f, g) = (-1)^{p(f)} (d_1(f) - 1)(d_1(g) - 1) f g,        p(C) = odd,

d_1 = the degree in the ODD arguments alone and fg the ordinary product.  It is
odd, so the deformation parameter hbar must be odd -- the odd Planck constant
M. Marinov asked about in 1977.
"""

from fractions import Fraction
from itertools import combinations


class Bn:
    def __init__(self, n):
        self.n = n

    # element = {(qexp tuple, xi tuple sorted): Fraction}
    def mono(self, qe, xi, c=1):
        return {(tuple(qe), tuple(sorted(xi))): Fraction(c)}

    @staticmethod
    def add(f, g, c=1):
        out = dict(f)
        for k, v in g.items():
            out[k] = out.get(k, Fraction(0)) + Fraction(c) * v
            if out[k] == 0:
                del out[k]
        return out

    @staticmethod
    def iszero(f):
        return all(v == 0 for v in f.values())

    def pfun(self, key):
        """parity of the monomial AS A FUNCTION."""
        return len(key[1]) % 2

    def pb(self, key):
        """parity in b(n)."""
        return (self.pfun(key) + 1) % 2

    def mul(self, f, g):
        out = {}
        for (qa, xa), ca in f.items():
            for (qb, xb), cb in g.items():
                if set(xa) & set(xb):
                    continue
                sgn = 1
                # sort xa + xb, counting transpositions
                merged = list(xa) + list(xb)
                for idx in range(len(xa)):
                    sgn *= (-1) ** sum(1 for y in xb if y < xa[idx])
                key = (tuple(a + b for a, b in zip(qa, qb)), tuple(sorted(merged)))
                out[key] = out.get(key, Fraction(0)) + sgn * ca * cb
        return {k: v for k, v in out.items() if v != 0}

    def dq(self, f, i):
        out = {}
        for (qa, xa), c in f.items():
            if qa[i] == 0:
                continue
            b = list(qa)
            e = b[i]
            b[i] -= 1
            k = (tuple(b), xa)
            out[k] = out.get(k, Fraction(0)) + e * c
        return {k: v for k, v in out.items() if v != 0}

    def dxi(self, f, i):
        """left derivative d/d xi_i."""
        out = {}
        for (qa, xa), c in f.items():
            if i not in xa:
                continue
            pos = xa.index(i)
            k = (qa, tuple(y for y in xa if y != i))
            out[k] = out.get(k, Fraction(0)) + ((-1) ** pos) * c
        return {k: v for k, v in out.items() if v != 0}

    def bracket(self, f, g):
        out = {}
        for i in range(self.n):
            out = self.add(out, self.mul(self.dq(f, i), self.dxi(g, i)))
        # the second sum needs the parity of each monomial of f
        for (qa, xa), c in f.items():
            fm = {(qa, xa): c}
            s = (-1) ** self.pfun((qa, xa))
            for i in range(self.n):
                out = self.add(out, self.mul(self.dxi(fm, i), self.dq(g, i)), s)
        return {k: v for k, v in out.items() if v != 0}

    # -- the quantizing cocycle -------------------------------------------
    def d1(self, key):
        return len(key[1])

    def delta(self, f):
        """the outer derivation  f |-> (d_1(f) - 1) f  (C. Roger's 1-cocycle)."""
        return {k: (self.d1(k) - 1) * v for k, v in f.items() if (self.d1(k) - 1) * v != 0}

    def C(self, f, g):
        """C(f,g) = (-1)^{p(f)} (d1(f)-1)(d1(g)-1) f g."""
        out = {}
        for kf, cf in f.items():
            for kg, cg in g.items():
                s = ((-1) ** self.pfun(kf)) * (self.d1(kf) - 1) * (self.d1(kg) - 1)
                if s == 0:
                    continue
                out = self.add(out, self.mul({kf: cf}, {kg: cg}), s)
        return {k: v for k, v in out.items() if v != 0}

    def basis(self, maxdeg):
        out = []
        for k in range(self.n + 1):
            for xi in combinations(range(self.n), k):
                for qe in self._exps(maxdeg):
                    if sum(qe) + k <= maxdeg:
                        out.append((tuple(qe), tuple(xi)))
        return out

    def _exps(self, maxdeg):
        if self.n == 0:
            yield ()
            return
        def rec(k, left):
            if k == self.n - 1:
                for e in range(left + 1):
                    yield (e,)
                return
            for e in range(left + 1):
                for rest in rec(k + 1, left - e):
                    yield (e,) + rest
        yield from rec(0, maxdeg)
