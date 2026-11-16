#!/usr/bin/env python3
r"""Polynomial vector fields on a superspace, over Q, in exact arithmetic.

The engine behind the rigidity computations for the exceptional simple
vectorial Lie superalgebras.

Coordinates:  x_0, ..., x_{m-1}   (even),  th_0, ..., th_{n-1}  (odd).
A monomial is a pair (exps, odds) with exps a tuple of m non-negative integers
and odds a strictly increasing tuple of odd indices; it denotes

    x^exps * th_{odds[0]} * ... * th_{odds[-1]}        (odd factors in this order).

A function is a dict {monomial: Fraction}; a vector field is a dict
{(gen, monomial): Fraction} standing for sum of  coeff * monomial * d/d(gen),
where gen is 0..m-1 for the even coordinates and m..m+n-1 for the odd ones.

Everything is graded by an arbitrary weight vector on the coordinates (used for
the Z-grading of a W-regrading) and by the weights of a torus.
"""
from fractions import Fraction

# --------------------------------------------------------------- the ambient


class Space:
    """The superspace with its coordinates, their degrees and torus weights."""

    def __init__(self, nev, nod, deg=None, wt=None):
        self.nev, self.nod = nev, nod
        self.n = nev + nod
        # degree of each coordinate for the Z-grading of the vector fields
        self.deg = list(deg) if deg is not None else [1] * self.n
        # torus weight of each coordinate (a tuple), used to split the complex
        self.wt = [tuple(w) for w in wt] if wt is not None else [()] * self.n

    def par_gen(self, g):
        return 0 if g < self.nev else 1

    # ------------------------------------------------------------ monomials
    def one(self):
        return ((0,) * self.nev, ())

    def par_mono(self, m):
        return len(m[1]) % 2

    def deg_mono(self, m):
        a, S = m
        d = sum(a[i] * self.deg[i] for i in range(self.nev))
        d += sum(self.deg[self.nev + j] for j in S)
        return d

    def wt_mono(self, m):
        a, S = m
        w = [0] * len(self.wt[0]) if self.wt and self.wt[0] else []
        for i in range(self.nev):
            if a[i]:
                w = [x + a[i] * y for x, y in zip(w, self.wt[i])]
        for j in S:
            w = [x + y for x, y in zip(w, self.wt[self.nev + j])]
        return tuple(w)

    def mono_mul(self, m1, m2):
        """Product of two monomials; returns (sign, monomial) or None."""
        a1, S1 = m1
        a2, S2 = m2
        if set(S1) & set(S2):
            return None
        merged = list(S1) + list(S2)
        # sign of the permutation sorting `merged`
        sign = 1
        arr = merged[:]
        for i in range(len(arr)):
            for j in range(len(arr) - 1, i, -1):
                if arr[j] < arr[j - 1]:
                    arr[j], arr[j - 1] = arr[j - 1], arr[j]
                    sign = -sign
        a = tuple(x + y for x, y in zip(a1, a2))
        return sign, (a, tuple(arr))

    def deriv(self, g, m):
        """d/d(gen g) applied to the monomial m; returns (coeff, monomial) or None."""
        a, S = m
        if g < self.nev:
            if a[g] == 0:
                return None
            b = list(a)
            b[g] -= 1
            return a[g], (tuple(b), S)
        j = g - self.nev
        if j not in S:
            return None
        pos = S.index(j)
        sign = -1 if pos % 2 else 1
        T = tuple(k for k in S if k != j)
        return sign, (a, T)

    # -------------------------------------------------------- vector fields
    def vf_par(self, D):
        """Parity of a (homogeneous) vector field."""
        for (g, m), c in D.items():
            if c:
                return (self.par_mono(m) + self.par_gen(g)) % 2
        return 0

    def vf_deg(self, D):
        for (g, m), c in D.items():
            if c:
                return self.deg_mono(m) - self.deg[g]
        return None

    def vf_wt(self, D):
        for (g, m), c in D.items():
            if c:
                return tuple(x - y for x, y in zip(self.wt_mono(m), self.wt[g]))
        return None

    def bracket(self, D1, D2):
        """[D1, D2] for homogeneous D1, D2."""
        p1, p2 = self.vf_par(D1), self.vf_par(D2)
        out = {}
        for (i, m1), c1 in D1.items():
            if not c1:
                continue
            for (j, m2), c2 in D2.items():
                if not c2:
                    continue
                # first term:  m1 * d_i(m2) * d_j
                r = self.deriv(i, m2)
                if r is not None:
                    co, mm = r
                    pr = self.mono_mul(m1, mm)
                    if pr is not None:
                        s, mono = pr
                        k = (j, mono)
                        out[k] = out.get(k, 0) + c1 * c2 * co * s
                # second term: -(-1)^{p1 p2} m2 * d_j(m1) * d_i
                r = self.deriv(j, m1)
                if r is not None:
                    co, mm = r
                    pr = self.mono_mul(m2, mm)
                    if pr is not None:
                        s, mono = pr
                        k = (i, mono)
                        sg = -1 if (p1 * p2) % 2 == 0 else 1
                        out[k] = out.get(k, 0) + sg * c1 * c2 * co * s
        return {k: v for k, v in out.items() if v}

    # ------------------------------------------------------------- monomials
    def monomials(self, deg):
        """All monomials of the given degree."""
        res = []
        from itertools import combinations

        for k in range(self.nod + 1):
            for S in combinations(range(self.nod), k):
                d0 = sum(self.deg[self.nev + j] for j in S)
                rest = deg - d0
                if rest < 0:
                    continue
                for a in self._evenmons(rest):
                    res.append((a, S))
        return res

    def _evenmons(self, deg):
        """All exponent tuples of the given weighted degree in the even coordinates."""
        out = []

        def rec(i, remaining, acc):
            if i == self.nev:
                if remaining == 0:
                    out.append(tuple(acc))
                return
            d = self.deg[i]
            k = 0
            while k * d <= remaining:
                rec(i + 1, remaining - k * d, acc + [k])
                k += 1

        rec(0, deg, [])
        return out

    def field_basis(self, deg):
        """All monomial vector fields of the given degree."""
        res = []
        for g in range(self.n):
            for m in self.monomials(deg + self.deg[g]):
                res.append({(g, m): Fraction(1)})
        return res


# ------------------------------------------------------------ linear algebra


def vec_of(D, index):
    v = [Fraction(0)] * len(index)
    for k, c in D.items():
        v[index[k]] = Fraction(c)
    return v


def build_index(fields):
    idx = {}
    for D in fields:
        for k in D:
            if k not in idx:
                idx[k] = len(idx)
    return idx
