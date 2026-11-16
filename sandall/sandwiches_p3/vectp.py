#!/usr/bin/env python3
r"""
vectp.py --- the divided power algebra O(n; \One) and vect(n; \One|0) over F_p,
for any p >= 2 and any n, built from the definition:

        x^(a) x^(b) = binom(a+b, a) x^(a+b)   (0 if some a_i + b_i > p - 1),
        d_i x^(a) = x^(a - e_i),
        [f d_i, g d_j] = f (d_i g) d_j - g (d_j f) d_i,
        Div(sum c_i d_i) = sum d_i c_i.

Used (i) to check the criterion (fs) of the paper --- for
s = sum c_i d_i in vect(n; \One|0),

        (ad_s)^2 = 0  <=>  c_i c_j = 0, c_k d_i c_j = 0, c_k d_i(Div s) = 0
                           for all 1 <= i, j, k <= n,

and (ii) as the ambient algebra of the Cartan series.
"""

import itertools
import numpy as np

from sandp import Alg


class O:
    """O(n; \\One) over F_p: divided powers, every indeterminate of height 1."""

    def __init__(self, p, n):
        self.p, self.n = p, n
        self.mons = [tuple(a) for a in itertools.product(range(p), repeat=n)]
        self.idx = {m: i for i, m in enumerate(self.mons)}
        self.N = len(self.mons)
        self._mul = None

    def binom(self, a, b):
        r = 1
        for ai, bi in zip(a, b):
            if ai + bi > self.p - 1:
                return 0
            # ordinary binomial, then mod p (all entries < p, so no Lucas needed)
            num, den = 1, 1
            for t in range(ai):
                num *= (ai + bi - t)
                den *= (t + 1)
            r = r * (num // den) % self.p
        return r

    def mul(self):
        if self._mul is None:
            M = np.zeros((self.N, self.N, self.N), dtype=np.int64)
            for i, a in enumerate(self.mons):
                for j, b in enumerate(self.mons):
                    c = self.binom(a, b)
                    if c:
                        M[i, j, self.idx[tuple(x + y for x, y in zip(a, b))]] = c
            self._mul = M
        return self._mul

    def prod(self, f, g):
        return np.einsum("i,j,ijk->k", f % self.p, g % self.p, self.mul()) % self.p

    def diff(self, i, f):
        out = np.zeros(self.N, dtype=np.int64)
        for j, c in enumerate(f):
            if c % self.p:
                a = self.mons[j]
                if a[i] > 0:
                    b = list(a)
                    b[i] -= 1
                    out[self.idx[tuple(b)]] = (out[self.idx[tuple(b)]] + c) % self.p
        return out % self.p


def vect(p, n):
    """vect(n; \\One|0) as an Alg; the basis is (monomial, direction)."""
    o = O(p, n)
    basis = [(m, i) for i in range(n) for m in o.mons]
    pos = {b: k for k, b in enumerate(basis)}
    dim = len(basis)

    def field(vecs):
        """vecs: list of n coefficient vectors c_i -> a dim-vector."""
        out = np.zeros(dim, dtype=np.int64)
        for i, c in enumerate(vecs):
            for j, co in enumerate(c):
                if co % p:
                    out[pos[(o.mons[j], i)]] = co % p
        return out

    def coeffs(v):
        cs = [np.zeros(o.N, dtype=np.int64) for _ in range(n)]
        for k, co in enumerate(v):
            if co % p:
                m, i = basis[k]
                cs[i][o.idx[m]] = co % p
        return cs

    C = np.zeros((dim, dim, dim), dtype=np.int64)
    for a, (ma, i) in enumerate(basis):
        f = np.zeros(o.N, dtype=np.int64)
        f[o.idx[ma]] = 1
        for b, (mb, j) in enumerate(basis):
            g = np.zeros(o.N, dtype=np.int64)
            g[o.idx[mb]] = 1
            t1 = o.prod(f, o.diff(i, g))       # f (d_i g) d_j
            t2 = o.prod(g, o.diff(j, f))       # g (d_j f) d_i
            res = [np.zeros(o.N, dtype=np.int64) for _ in range(n)]
            res[j] = (res[j] + t1) % p
            res[i] = (res[i] - t2) % p
            C[a, b] = field(res)

    deg = [sum(m) - 1 for (m, i) in basis]
    wt = [tuple(mm - (1 if t == i else 0) for t, mm in enumerate(m))
          for (m, i) in basis]
    g = Alg(p, C, deg, wt, name="vect(%d; 1)" % n)
    g.O, g.field, g.coeffs, g.basis_lab = o, field, coeffs, basis
    return g


def div(g, v):
    """Div of a vector field given in the basis of vect(n; \\One|0)."""
    o, cs = g.O, g.coeffs(v)
    out = np.zeros(o.N, dtype=np.int64)
    for i, c in enumerate(cs):
        out = (out + o.diff(i, c)) % g.p
    return out


def fs_rhs(g, v):
    """the right-hand side of (fs): c_i c_j = 0, c_k d_i c_j = 0,
    c_k d_i (Div s) = 0 for all i, j, k."""
    o, p = g.O, g.p
    cs = g.coeffs(v)
    n = len(cs)
    ds = div(g, v)
    for i in range(n):
        for j in range(n):
            if o.prod(cs[i], cs[j]).any():
                return False
            for k in range(n):
                if o.prod(cs[k], o.diff(i, cs[j])).any():
                    return False
        for k in range(n):
            if o.prod(cs[k], o.diff(i, ds)).any():
                return False
    return True
