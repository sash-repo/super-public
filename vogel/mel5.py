#!/usr/bin/env python3
"""The Melikyan algebra at p=5 as a prolong inside the contact algebra k(5; N),
and its H^2(g_-; g) with the g_0-invariants (filtered deformations, Cheng--Kac).

Coordinates: t (degree 2) and y_1..y_4 (degree 1); y_k is dual to x^{(k)} in
O(1;1)/K, the g_{-1} of the contact grading.  Contact bracket with a constant
Poisson tensor pi (computed, not assumed: the vect(1;1)-invariant form on O(1;1)/K):

    {f,g} = (2-E)f d_t g - d_t f (2-E)g + sum pi_ij d_i f d_j g,  E = sum y_i d_i.

Divided powers: variable v has exponents 0..LIM[v]-1, LIM = p^{N_v}.
"""
import sys, itertools
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace as _ns, rank as _rk, rref as _rref

P = 5


def ns(M, n):
    if not M:
        return [[1 if i == j else 0 for i in range(n)] for j in range(n)]
    return [[int(x) % P for x in r] for r in _ns(M, P)]


def rk(M):
    return _rk(M, P) if M else 0


def binom(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


class Contact:
    """k(5; N): monomials are 5-tuples (a_t, a_1..a_4)."""

    def __init__(self, pi, lim):
        self.pi = pi
        self.lim = lim
        self.deg = lambda m: 2 * m[0] + sum(m[1:])

    def mul(self, f, g):
        out = {}
        for m1, u in f.items():
            for m2, v in g.items():
                m = tuple(a + b for a, b in zip(m1, m2))
                if any(a >= L for a, L in zip(m, self.lim)):
                    continue
                c = u * v
                for a, a1 in zip(m, m1):
                    c *= binom(a, a1)
                c %= P
                if c:
                    out[m] = (out.get(m, 0) + c) % P
        return {k: v for k, v in out.items() if v}

    @staticmethod
    def d(f, i):
        out = {}
        for m, u in f.items():
            if m[i]:
                n = list(m); n[i] -= 1; n = tuple(n)
                out[n] = (out.get(n, 0) + u) % P
        return {k: v for k, v in out.items() if v}

    @staticmethod
    def lin(*terms):
        out = {}
        for c, f in terms:
            for m, u in f.items():
                out[m] = (out.get(m, 0) + c * u) % P
        return {k: v for k, v in out.items() if v}

    @staticmethod
    def twoE(f):
        return {m: u * (2 - sum(m[1:])) % P for m, u in f.items() if u * (2 - sum(m[1:])) % P}

    def br(self, f, g):
        terms = [(1, self.mul(self.twoE(f), self.d(g, 0))),
                 (-1, self.mul(self.d(f, 0), self.twoE(g)))]
        for i in range(4):
            for j in range(4):
                if self.pi[i][j]:
                    terms.append((self.pi[i][j], self.mul(self.d(f, i + 1), self.d(g, j + 1))))
        return self.lin(*terms)

    def monos(self, k):
        """monomials of generating-function degree k (field degree k-2)."""
        if not hasattr(self, "_bydeg"):
            self._bydeg = {}
            for m in itertools.product(*[range(L) for L in self.lim]):
                self._bydeg.setdefault(self.deg(m), []).append(m)
        return self._bydeg.get(k, [])


def mono(*e):
    return {tuple(e): 1}


# ------------------------------------------------ g_{-1} = O(1;1)/K as vect(1;1)-module
def vect11_on_quot():
    """matrices of x^{(k)} d (k=0..4) on span(x^{(1)},..,x^{(4)}), columns = images."""
    mats = []
    for k in range(5):
        A = [[0] * 4 for _ in range(4)]
        for j in range(1, 5):          # x^{(k)} d x^{(j)} = binom(k+j-1,k) x^{(k+j-1)}
            n = k + j - 1
            if 1 <= n <= 4:
                A[n - 1][j - 1] = binom(n, k) % P
        mats.append(A)
    return mats


def invariant_forms(mats):
    """antisymmetric B on g_{-1} with B(Au,v)+B(u,Av)=0: A^T B + B A = 0."""
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    rows = []
    for A in mats:
        for r in range(4):
            for s in range(4):
                row = [0] * 6
                for idx, (i, j) in enumerate(pairs):
                    # B = E_ij - E_ji ; (A^T B + B A)_{rs}
                    B = [[0] * 4 for _ in range(4)]
                    B[i][j], B[j][i] = 1, -1
                    v = sum(A[k][r] * B[k][s] for k in range(4)) + sum(B[r][k] * A[k][s] for k in range(4))
                    row[idx] = v % P
                rows.append(row)
    sols = ns(rows, 6)
    out = []
    for s in sols:
        B = [[0] * 4 for _ in range(4)]
        for c, (i, j) in zip(s, pairs):
            B[i][j], B[j][i] = c % P, -c % P
        out.append(B)
    return out


def inv4(M):
    """inverse mod P of a 4x4 matrix."""
    n = 4
    A = [r[:] + [1 if i == j else 0 for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        piv = next(r for r in range(c, n) if A[r][c] % P)
        A[c], A[piv] = A[piv], A[c]
        iv = pow(A[c][c], P - 2, P)
        A[c] = [x * iv % P for x in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]
                A[r] = [(x - f * y) % P for x, y in zip(A[r], A[c])]
    return [r[n:] for r in A]


def quad_hamiltonian(K, A):
    """quadratic H in y with {H, y_j} = sum_n A[n][j] y_n, or None."""
    qm = [m for m in K.monos(2) if m[0] == 0]
    rows, rhs = [], []
    ys = [mono(0, *[1 if i == j else 0 for i in range(4)]) for j in range(4)]
    cols = []
    for m in qm:
        cols.append([K.br({m: 1}, y) for y in ys])
    lin = [m for m in K.monos(1)]
    M = []
    for j in range(4):
        target = {}
        for n in range(4):
            if A[n][j]:
                target[lin_index(n)] = A[n][j]
        for lm in lin:
            M.append([cols[c][j].get(lm, 0) for c in range(len(qm))] + [(-target.get(lm, 0)) % P])
    sol = ns(M, len(qm) + 1)
    for s in sol:
        if s[-1] % P:
            iv = pow(s[-1], P - 2, P)
            return {m: c * iv % P for m, c in zip(qm, s[:-1]) if c * iv % P}
    return None


def lin_index(n):
    return tuple([0] + [1 if i == n else 0 for i in range(4)])


def setup(lim=(5, 5, 5, 5, 5)):
    mats = vect11_on_quot()
    forms = invariant_forms(mats)
    return mats, forms


class Graded:
    """graded subalgebra of k(5;N) given by generating functions, per degree."""

    def __init__(self, K, comps):
        self.K = K
        self.comps = comps                    # degree(field) -> list of polys
        self.basis, self.fdeg = [], []
        for d in sorted(comps):
            for f in comps[d]:
                self.basis.append(f); self.fdeg.append(d)
        self.n = len(self.basis)
        self._coord = {}
        for d in comps:
            ms = sorted({m for f in comps[d] for m in f})
            self._coord[d] = (ms, [[f.get(m, 0) for m in ms] for f in comps[d]])
        self.start = {}
        i = 0
        for d in sorted(comps):
            self.start[d] = i; i += len(comps[d])

    def coords(self, f, d):
        """coordinates of polynomial f (field degree d) in the basis."""
        if not f:
            return {}
        if d not in self.comps:
            raise ValueError("bracket leaves the algebra (degree %d)" % d)
        ms, rows = self._coord[d]
        if any(m not in set(ms) for m in f):
            raise ValueError("not in the algebra")
        k = len(rows)
        # solve sum c_i rows_i = f
        M = [[rows[i][j] for i in range(k)] + [(-f.get(m, 0)) % P] for j, m in enumerate(ms)]
        for s in ns(M, k + 1):
            if s[-1] % P:
                iv = pow(s[-1], P - 2, P)
                return {self.start[d] + i: c * iv % P for i, c in enumerate(s[:-1]) if c * iv % P}
        raise ValueError("not in the algebra")


def prolong(K, gm2, gm1, g0, kmax=40):
    comps = {-2: gm2, -1: gm1, 0: g0}
    k = 1
    while k <= kmax:
        cand = K.monos(k + 2)
        if not cand:
            break
        prev = comps[k - 1]
        pms = sorted({m for f in prev for m in f})
        # condition: {f, y} in span(prev) for y in g_{-1}
        # unknowns: coefficients of f (len cand) and of the images (len(prev) per y)
        nprev = len(prev)
        ncol = len(cand) + nprev * len(gm1)
        rows = {}
        for ci, m in enumerate(cand):
            for yi, y in enumerate(gm1):
                for mm, c in K.br({m: 1}, y).items():
                    rows.setdefault((yi, mm), [0] * ncol)[ci] = c
        for yi in range(len(gm1)):
            for pi_, f in enumerate(prev):
                for mm, c in f.items():
                    r = rows.setdefault((yi, mm), [0] * ncol)
                    r[len(cand) + yi * nprev + pi_] = (r[len(cand) + yi * nprev + pi_] - c) % P
        sol = ns(list(rows.values()), ncol)
        fs = [[s[:len(cand)] for s in sol]]
        R = _rref(fs[0], P)[0] if fs[0] else []
        basis = []
        for r in R:
            r = [int(x) % P for x in r]
            if any(r):
                basis.append({m: c for m, c in zip(cand, r) if c})
        if not basis:
            break
        comps[k] = basis
        k += 1
    return comps
