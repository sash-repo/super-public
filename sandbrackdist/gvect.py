#!/usr/bin/env python3
"""Generic vectorial Lie algebra vect(n; 1) over F_3 in divided powers, plus the
F_3 linear algebra (numpy, int8) the prolongations and the cohomology need.

O(n; 1) = span of divided-power monomials u_1^(a_1)...u_n^(a_n), 0 <= a_i <= p-1.
A vector field is a dict {(mon, i): coeff} standing for sum coeff * mon * d/du_i.

Each coordinate u_a carries an integer Z-degree deg[a] and a weight wt[a] in Z^r
(the weight of the maximal torus of g_0).  Then

    deg(f d_a) = deg f - deg[a],      wt(f d_a) = wt(f) - wt[a].
"""
import numpy as np
from itertools import product as iproduct

P = 3


# --------------------------------------------------------------- F_3 linear algebra
def rref(M):
    """row-reduce M (numpy int8, mod 3) in place; return (R, pivots)."""
    if M.size == 0:
        return M.copy(), []
    A = (np.asarray(M, dtype=np.int64) % P).astype(np.int64)
    rows, cols = A.shape
    piv, r = [], 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        inv = pow(int(A[r, c]), P - 2, P)
        A[r] = (A[r] * inv) % P
        col = A[:, c].copy()
        col[r] = 0
        nzr = np.nonzero(col)[0]
        if nzr.size:
            A[nzr] = (A[nzr] - np.outer(col[nzr], A[r])) % P
        piv.append(c)
        r += 1
    return A[:r], piv


def rank(M):
    return len(rref(M)[1])


def nullspace(M, ncols=None):
    """basis (rows) of {v : M v = 0}."""
    if M is None or (hasattr(M, "size") and M.size == 0) or len(M) == 0:
        n = ncols
        return np.eye(n, dtype=np.int64) % P
    A = np.asarray(M, dtype=np.int64) % P
    n = A.shape[1] if ncols is None else ncols
    R, piv = rref(A)
    free = [c for c in range(n) if c not in piv]
    out = np.zeros((len(free), n), dtype=np.int64)
    for k, f in enumerate(free):
        out[k, f] = 1
        for i, c in enumerate(piv):
            out[k, c] = (-R[i, f]) % P
    return out


def left_nullspace(M, nrows=None):
    """basis (rows) of {v : v M = 0}."""
    A = np.asarray(M, dtype=np.int64) % P
    if A.size == 0:
        return np.eye(nrows if nrows is not None else A.shape[0], dtype=np.int64)
    return nullspace(A.T, A.shape[0])


def solve_in_span(basis, vecs):
    """coordinates of each row of `vecs` in the span of the rows of `basis`
    (which must be independent).  Raises if some row is not in the span."""
    B = np.asarray(basis, dtype=np.int64) % P
    R, piv = rref(B)
    out = []
    for v in np.asarray(vecs, dtype=np.int64) % P:
        w = v.copy() % P
        co = np.zeros(len(R), dtype=np.int64)
        for i, c in enumerate(piv):
            if w[c]:
                co[i] = w[c] % P
                w = (w - co[i] * R[i]) % P
        if w.any():
            raise ValueError("vector not in the span")
        # co are coordinates w.r.t. R, convert to coordinates w.r.t. B
        out.append(co)
    # change of basis R = T B  ->  coordinates w.r.t. B are co T
    T = np.zeros((len(R), len(B)), dtype=np.int64)
    Bx = B.copy()
    aug = np.concatenate([Bx, np.eye(len(B), dtype=np.int64)], axis=1)
    Ra, pv = rref(aug)
    T = Ra[:, B.shape[1]:]
    return (np.asarray(out) @ T) % P


# --------------------------------------------------------------- divided powers
def binom(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


class Vect:
    """vect(n; 1) over F_3 with the given degrees and weights of the coordinates."""

    def __init__(self, degs, wts, names=None):
        self.n = len(degs)
        self.degs = list(degs)
        self.wts = [tuple(w) for w in wts]
        self.names = list(names) if names else ["u%d" % i for i in range(self.n)]
        self.mons = [m for m in iproduct(*[range(P)] * self.n)]
        self.mdeg = {m: sum(a * d for a, d in zip(m, self.degs)) for m in self.mons}
        self.mwt = {m: tuple(sum(a * w[k] for a, w in zip(m, self.wts))
                             for k in range(len(self.wts[0]))) for m in self.mons}

    # ---- polynomials
    def pmul(self, f, g):
        out = {}
        for m1, u in f.items():
            for m2, v in g.items():
                m, co = [], u * v
                ok = True
                for a, b in zip(m1, m2):
                    s = a + b
                    if s >= P:
                        ok = False
                        break
                    co *= binom(s, a)
                    m.append(s)
                if not ok:
                    continue
                co %= P
                if co:
                    m = tuple(m)
                    out[m] = (out.get(m, 0) + co) % P
        return {k: v for k, v in out.items() if v}

    def dpart(self, f, a):
        out = {}
        for m, u in f.items():
            if m[a] == 0:
                continue
            k = list(m)
            k[a] -= 1
            k = tuple(k)
            out[k] = (out.get(k, 0) + u) % P
        return {k: v for k, v in out.items() if v}

    # ---- vector fields
    def deg(self, key):
        m, i = key
        return self.mdeg[m] - self.degs[i]

    def wt(self, key):
        m, i = key
        w = self.mwt[m]
        return tuple(a - b for a, b in zip(w, self.wts[i]))

    def fdeg(self, D):
        ds = {self.deg(k) for k in D}
        assert len(ds) == 1, "field is not homogeneous"
        return ds.pop()

    def fwt(self, D):
        ws = {self.wt(k) for k in D}
        assert len(ws) == 1, "field is not weight-homogeneous"
        return ws.pop()

    def bracket(self, D, E):
        out = {}

        def add(m, i, co):
            co %= P
            if co:
                k = (m, i)
                out[k] = (out.get(k, 0) + co) % P

        for (m1, i), c1 in D.items():
            for (m2, j), c2 in E.items():
                c = c1 * c2 % P
                if not c:
                    continue
                for mm, cc in self.pmul({m1: 1}, self.dpart({m2: 1}, i)).items():
                    add(mm, j, c * cc)
                for mm, cc in self.pmul({m2: 1}, self.dpart({m1: 1}, j)).items():
                    add(mm, i, -c * cc)
        return {k: v for k, v in out.items() if v}

    def basis_of_degree(self, k):
        return [(m, i) for m in self.mons for i in range(self.n)
                if self.mdeg[m] - self.degs[i] == k]

    def show(self, D):
        terms = []
        for (m, i), c in sorted(D.items()):
            s = "" if c == 1 else str(c)
            for a, e in zip(self.names, m):
                if e == 1:
                    s += a
                elif e == 2:
                    s += "%s^(2)" % a
            terms.append("%sd_%s" % (s if s else "", self.names[i]))
        return " + ".join(terms) if terms else "0"
