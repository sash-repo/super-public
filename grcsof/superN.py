#!/usr/bin/env python3
"""Superfunctions C[t] (x) Lambda[y_1,...,y_n] and the contact superalgebra
k(1|n) for an arbitrary symmetric Gram matrix B of the odd coordinates, in the
conventions of [BLS1] = Bouarroudj-Leites-Shchepochkina, "Analogs of Bol
operators on superstrings", IJAC 32 (2022) 807-835, arXiv:2110.10504.

Contact form            alpha = dt + sum_{i,j} B_{ij} y_i d y_j,  B = B^T,
contact fields          K_f = (2-E)(f) d_t - H_f + d_t(f) E,   E = sum y_i d_i,
                        H_f = -(-1)^{p(f)} sum_{i,j} C_{ij} d_i(f) d_j,  C = B^{-1},
contact bracket         {f,g} = (2-E)(f) d_t(g) - d_t(f) (2-E)(g) - {f,g}_{P.b.},
                        {f,g}_{P.b.} = -(-1)^{p(f)} sum C_{ij} d_i(f) d_j(g),
invariant derivatives   D_i = sum_j B_{ij} y_j d_t + d_i,   [D_i,D_j] = 2 B_{ij} d_t,
weighted densities      F_lam = F alpha^{lam/2},
                        L_{K_f}(phi) = K_f(phi) + lam d_t(f) phi.

For n = 2 and B = antidiag(1,1) this is exactly super12.py/k12.py (regression
test in check_superN.py), i.e. alpha_1 = dt + xi d(eta) + eta d(xi).

Monomials are pairs (a, mask): t^a * y^mask, mask a bitmask with y_1 the lowest
bit and the factors written in increasing order.  Coefficients are anything
that supports +, *, and comparison with 0 (Fraction for speed, sympy for
symbolic weights).
"""
from fractions import Fraction

# --------------------------------------------------------------- book-keeping


def clean(u):
    return {k: v for k, v in u.items() if v != 0}


def add(*us):
    out = {}
    for u in us:
        for k, v in u.items():
            out[k] = out.get(k, 0) + v
    return clean(out)


def smul(c, u):
    if c == 0:
        return {}
    return clean({k: c * v for k, v in u.items()})


def popcount(m):
    return bin(m).count('1')


def par_mon(m):
    return popcount(m[1]) % 2


def parity(u):
    ps = {par_mon(m) for m in u}
    assert len(ps) <= 1, "inhomogeneous parity"
    return ps.pop() if ps else 0


def mono(a=0, mask=0, c=1):
    return {(a, mask): c}


ONE = mono()


def _mulsign(m1, m2):
    """sign of y^m1 . y^m2 when the result is re-ordered increasingly."""
    s = 0
    for j in range(m2.bit_length()):
        if m2 >> j & 1:
            # number of generators of m1 that stand to the right of y_j
            s += popcount(m1 >> (j + 1))
    return -1 if s % 2 else 1


def mul(u, v):
    out = {}
    for (a1, m1), c1 in u.items():
        for (a2, m2), c2 in v.items():
            if m1 & m2:
                continue
            k = (a1 + a2, m1 | m2)
            out[k] = out.get(k, 0) + _mulsign(m1, m2) * c1 * c2
    return clean(out)


def d_t(u):
    out = {}
    for (a, m), c in u.items():
        if a:
            out[(a - 1, m)] = out.get((a - 1, m), 0) + a * c
    return clean(out)


def d_odd(i):
    """left derivative d/d y_i  (i = 0,...,n-1)."""
    bit = 1 << i

    def r(u):
        out = {}
        for (a, m), c in u.items():
            if m & bit:
                s = -1 if popcount(m & (bit - 1)) % 2 else 1
                k = (a, m ^ bit)
                out[k] = out.get(k, 0) + s * c
        return clean(out)
    return r


def E_odd(u):
    return clean({k: popcount(k[1]) * v for k, v in u.items()})


def compose(*ops):
    def r(u):
        for o in reversed(ops):
            u = o(u)
        return u
    return r


def dtn(k):
    def r(u):
        for _ in range(k):
            u = d_t(u)
        return u
    return r


# ------------------------------------------------------------------- k(1|n)

class Contact:
    """k(1|n) for the contact form alpha = dt + sum B_{ij} y_i dy_j."""

    def __init__(self, n, B, names=None):
        self.n = n
        self.B = [[Fraction(x) for x in row] for row in B]
        self.C = self._inverse(self.B)
        self.names = names or [f"y{i+1}" for i in range(n)]
        self.d = [d_odd(i) for i in range(n)]

    @staticmethod
    def _inverse(B):
        n = len(B)
        M = [row[:] + [Fraction(int(i == j)) for j in range(n)]
             for i, row in enumerate(B)]
        for col in range(n):
            piv = next(r for r in range(col, n) if M[r][col] != 0)
            M[col], M[piv] = M[piv], M[col]
            pv = M[col][col]
            M[col] = [x / pv for x in M[col]]
            for r in range(n):
                if r != col and M[r][col] != 0:
                    f = M[r][col]
                    M[r] = [x - f * y for x, y in zip(M[r], M[col])]
        return [row[n:] for row in M]

    # --- the field K_f
    def K(self, f):
        p = parity(f)
        sgn = -1 if p else 1                       # -(-1)^{p(f)} . (-1) = +(-1)^{p(f)}
        two_minus_E = add(smul(2, f), smul(-1, E_odd(f)))
        dtf = d_t(f)
        df = [self.d[i](f) for i in range(self.n)]

        def r(u):
            out = mul(two_minus_E, d_t(u))
            for i in range(self.n):
                if not df[i]:
                    continue
                for j in range(self.n):
                    cij = self.C[i][j]
                    if cij:
                        out = add(out, smul(sgn * cij, mul(df[i], self.d[j](u))))
            out = add(out, mul(dtf, E_odd(u)))
            return out
        return r

    # --- the contact bracket {f,g}
    def kb(self, f, g):
        p = parity(f)
        sgn = -1 if p else 1
        a = mul(add(smul(2, f), smul(-1, E_odd(f))), d_t(g))
        b = mul(d_t(f), add(smul(2, g), smul(-1, E_odd(g))))
        pb = {}
        for i in range(self.n):
            dif = self.d[i](f)
            if not dif:
                continue
            for j in range(self.n):
                cij = self.C[i][j]
                if cij:
                    pb = add(pb, smul(-sgn * cij, mul(dif, self.d[j](g))))
        return add(a, smul(-1, b), smul(-1, pb))

    # --- the invariant derivatives D_i = sum_j B_{ij} y_j d_t + d_i
    def D(self, i):
        coef = {}
        for j in range(self.n):
            if self.B[i][j]:
                coef = add(coef, mono(0, 1 << j, self.B[i][j]))

        def r(u):
            return add(mul(coef, d_t(u)), self.d[i](u))
        return r

    def Dword(self, subset, k=0):
        """d_t^k D_{i1} ... D_{ir} for subset = (i1 < ... < ir)."""
        ops = [self.D(i) for i in subset] + [dtn(k)]
        return compose(*ops)

    # --- weighted densities F_lam = F alpha^{lam/2}
    def L(self, f, lam):
        Kf = self.K(f)
        c = smul(lam, d_t(f))

        def r(u):
            return add(Kf(u), mul(c, u))
        return r

    # --- test monomials
    def monos(self, tmax):
        return [(a, m) for a in range(tmax + 1) for m in range(1 << self.n)]

    def show(self, u):
        if not u:
            return "0"
        out = []
        for (a, m), c in sorted(u.items()):
            s = f"({c})"
            if a:
                s += f"*t^{a}" if a > 1 else "*t"
            for i in range(self.n):
                if m >> i & 1:
                    s += "*" + self.names[i]
            out.append(s)
        return " + ".join(out)
