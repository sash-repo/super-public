#!/usr/bin/env python3
r"""
sandp.py --- sandwiches, the sandwich subalgebra s(g) and its normaliser
n_g(s(g)) for a finite-dimensional Z-graded Lie algebra over F_p with p >= 3.

The point of the module is that the *whole* sandwich variety

        S := {s in g | (ad_s)^2 = 0}

never has to be computed.  S is a cone, Zariski-closed, and stable under
Aut(g); that is enough to sandwich the normaliser between two computable
subalgebras:

 1) LOWER BOUND.  Any finite set S' of sandwiches that we can exhibit
    generates a subalgebra s' contained in s(g), whence

            n_g(s(g)) subset n_g(s').

    Both s' and n_g(s') are plain linear algebra.

 2) UPPER BOUND.  Let x in g with (ad_x)^p = 0.  Then exp(ad_{tx}) is an
    automorphism for every t in K, so exp(ad_{tx})S = S; expanding in t and
    using that the span <S> is a subspace and K is infinite (Vandermonde),

            ad_x <S> subset <S>,      hence      ad_x s(g) subset s(g),

    ad_x being a derivation.  The same for ad_h with h in the Lie algebra of
    a torus of Aut(g) acting on g, since <S> is then a sum of weight spaces.
    So every subalgebra spanned by ad-nilpotent elements and by such tori
    lies in n_g(s(g)), whatever s(g) is.

Together: if L_0 is spanned by ad-nilpotent elements and torus elements, and
if n_g(s') = L_0 for an explicitly exhibited s', then n_g(s(g)) = L_0 --- with
no knowledge of S beyond the few sandwiches actually written down.

Everything is exact arithmetic over F_p.
"""

import numpy as np


# --------------------------------------------------------------- linear algebra
def rref(M, p):
    """row-reduced echelon form over F_p; returns (rows, pivots)."""
    M = np.array(M, dtype=np.int64) % p
    if M.size == 0:
        return np.zeros((0, M.shape[1] if M.ndim == 2 else 0), np.int64), []
    rows, piv, r = M.copy(), [], 0
    for c in range(rows.shape[1]):
        nz = [i for i in range(r, rows.shape[0]) if rows[i, c] % p]
        if not nz:
            continue
        i = nz[0]
        if i != r:
            rows[[r, i]] = rows[[i, r]]
        rows[r] = rows[r] * pow(int(rows[r, c]), p - 2, p) % p
        for i in range(rows.shape[0]):
            if i != r and rows[i, c] % p:
                rows[i] = (rows[i] - rows[i, c] * rows[r]) % p
        piv.append(c)
        r += 1
        if r == rows.shape[0]:
            break
    return rows[:r] % p, piv


def rank(M, p):
    return len(rref(M, p)[0])


def in_span(rows, piv, v, p):
    """is v in the span of the rref rows?"""
    v = np.array(v, dtype=np.int64) % p
    for i, c in enumerate(piv):
        if v[c]:
            v = (v - v[c] * rows[i]) % p
    return not v.any()


def nullspace(M, ncols, p):
    """basis of {x | M x = 0}, as rows."""
    if len(M) == 0:
        return np.eye(ncols, dtype=np.int64)
    R, piv = rref(M, p)
    free = [c for c in range(ncols) if c not in piv]
    out = []
    for f in free:
        v = np.zeros(ncols, dtype=np.int64)
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = (-R[i, f]) % p
        out.append(v)
    return np.array(out, dtype=np.int64) if out else np.zeros((0, ncols), np.int64)


# ------------------------------------------------------------------- the algebra
class Alg:
    """a Lie algebra over F_p with a fixed homogeneous basis.

    C[i, j] is the vector [b_i, b_j]; deg[i] the Z-degree; wt[i] a hashable
    label of the torus weight (or None)."""

    def __init__(self, p, C, deg, wt=None, name=""):
        self.p, self.name = p, name
        self.C = np.array(C, dtype=np.int64) % p
        self.n = self.C.shape[0]
        self.deg = list(deg)
        self.wt = list(wt) if wt is not None else [None] * self.n
        assert self.C.shape == (self.n, self.n, self.n)

    # -- basics
    def bracket(self, x, y):
        p = self.p
        return (np.einsum("i,j,ijk->k", np.asarray(x) % p,
                          np.asarray(y) % p, self.C) % p)

    def ad(self, x):
        """the matrix of ad_x: column j is [x, b_j]."""
        p = self.p
        M = np.einsum("i,ijk->kj", np.asarray(x) % p, self.C) % p
        return M

    def check_jacobi(self, sample=None):
        n, p = self.n, self.p
        idx = range(n)
        for i in idx:
            for j in idx:
                if ((self.C[i, j] + self.C[j, i]) % p).any():
                    return False, ("antisymmetry", i, j)
        for i in idx:
            for j in idx:
                for k in idx:
                    s = (self.bracket(self.C[i, j], np.eye(n, dtype=np.int64)[k])
                         + self.bracket(self.C[j, k], np.eye(n, dtype=np.int64)[i])
                         + self.bracket(self.C[k, i], np.eye(n, dtype=np.int64)[j])) % p
                    if s.any():
                        return False, ("Jacobi", i, j, k)
        return True, None

    # -- sandwiches
    def is_sandwich(self, x):
        M = self.ad(x)
        return not (M @ M % self.p).any()

    def blocks(self):
        """basis indices grouped by (deg, wt)."""
        b = {}
        for i in range(self.n):
            b.setdefault((self.deg[i], self.wt[i]), []).append(i)
        return b

    def homogeneous_sandwiches(self, maxdim=8, only_deg=None):
        """all F_p-points of S inside each (deg, wt)-block of dimension <= maxdim,
        one representative per line.  Returns a list of (key, vector)."""
        p, out = self.p, []
        for key, idx in sorted(self.blocks().items(), key=lambda t: (t[0][0], str(t[0][1]))):
            if only_deg is not None and key[0] not in only_deg:
                continue
            d = len(idx)
            if d > maxdim:
                continue
            # one representative per projective point: first non-zero entry = 1
            for lead in range(d):
                for tail in range(p ** (d - lead - 1)):
                    co = [0] * lead + [1]
                    t = tail
                    for _ in range(d - lead - 1):
                        co.append(t % p)
                        t //= p
                    v = np.zeros(self.n, dtype=np.int64)
                    for c, i in zip(co, idx):
                        v[i] = c
                    if self.is_sandwich(v):
                        out.append((key, v))
        return out

    # -- subalgebras
    def generated(self, vecs):
        """the Lie subalgebra generated by the given vectors, as rref rows."""
        p = self.p
        rows, piv = rref(list(vecs), p)
        while True:
            new = []
            for a in rows:
                for b in rows:
                    v = self.bracket(a, b)
                    if v.any() and not in_span(rows, piv, v, p):
                        new.append(v)
            if not new:
                return rows, piv
            rows, piv = rref(np.concatenate([rows, np.array(new)]), p)

    def normaliser(self, rows):
        """{x | [x, s] in span(rows) for every s in rows}."""
        p, n = self.p, self.n
        R, piv = rref(rows, p)
        # a complement functional basis: the rows of the projection onto g/span
        comp = [c for c in range(n) if c not in piv]
        eqs = []
        for s in R:
            M = self.ad_right(s)          # x |-> [x, s]
            # reduce each column mod span(R), keep the complement coordinates
            for c in comp:
                eqs.append(self._reduce_row(M, R, piv, c))
        E = np.array(eqs, dtype=np.int64) % p if eqs else np.zeros((0, n), np.int64)
        return rref(nullspace(E, n, p), p)

    def ad_right(self, s):
        """the matrix of x |-> [x, s]: column i is [b_i, s]."""
        p = self.p
        return np.einsum("j,ijk->ki", np.asarray(s) % p, self.C) % p

    def _reduce_row(self, M, R, piv, c):
        """the c-th coordinate, after reduction mod span(R), of x |-> M x,
        as a linear functional of x."""
        p = self.p
        # reduce M's columns modulo span(R): M' = (I - P) M
        Mr = M.copy() % p
        for i, pc in enumerate(piv):
            f = Mr[pc].copy()
            Mr = (Mr - np.outer(R[i], f)) % p
        return Mr[c] % p

    def is_maximal(self, rows):
        """is span(rows) a maximal subalgebra?  (enumerates the lines of g/h)"""
        p, n = self.p, self.n
        R, piv = rref(rows, p)
        comp = [c for c in range(n) if c not in piv]
        k = len(comp)
        if p ** k > 2 * 10 ** 6:
            return None
        for lead in range(k):
            for tail in range(p ** (k - lead - 1)):
                co = [0] * lead + [1]
                t = tail
                for _ in range(k - lead - 1):
                    co.append(t % p)
                    t //= p
                v = np.zeros(n, dtype=np.int64)
                for c, i in zip(co, comp):
                    v[i] = c
                big, _ = self.generated(list(R) + [v])
                if len(big) != n:
                    return False
        return True

    # -- the upper-bound certificate
    def ad_nilpotent_span(self, rows):
        """the span of those x in span(rows) with (ad_x)^p = 0, together with
        the diagonal torus elements (basis vectors of weight 0 acting
        semisimply).  Returns (rref rows, covered?)."""
        p = self.p
        R, piv = rref(rows, p)
        good = []
        k = len(R)
        if p ** k <= 3 * 10 ** 5:
            pts = []
            for lead in range(k):
                for tail in range(p ** (k - lead - 1)):
                    co = [0] * lead + [1]
                    t = tail
                    for _ in range(k - lead - 1):
                        co.append(t % p)
                        t //= p
                    pts.append(np.array(co, dtype=np.int64))
            for co in pts:
                v = (co @ R) % p
                M = self.ad(v)
                Mp = np.eye(self.n, dtype=np.int64)
                for _ in range(p):
                    Mp = Mp @ M % p
                if not Mp.any():
                    good.append(v)
        G = rref(good, p)[0] if good else np.zeros((0, self.n), np.int64)
        return G

    def semisimple_in(self, rows):
        """basis vectors inside span(rows) whose ad is diagonalisable over F_p
        (in practice: the weight-grading elements).  Detected by ad^p = ad."""
        p, out = self.p, []
        R, piv = rref(rows, p)
        for v in R:
            M = self.ad(v)
            Mp = M.copy()
            for _ in range(p - 1):
                Mp = Mp @ M % p
            if not ((Mp - M) % p).any():
                out.append(v)
        return rref(out, p)[0] if out else np.zeros((0, self.n), np.int64)
