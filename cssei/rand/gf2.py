#!/usr/bin/env python3
"""GF(2) linear algebra on numpy uint8 0/1 arrays; products through float32 BLAS
(exact: every sum is < 2^24)."""
import numpy as np

def mm(A, B):
    """A @ B mod 2."""
    return (np.matmul(A.astype(np.float32), B.astype(np.float32)).astype(np.int64) & 1).astype(np.uint8)

def rref(M):
    """Reduced row echelon form of M (k x L); returns (R, pivots)."""
    M = (M.copy() & 1).astype(np.uint8)
    k, L = M.shape
    piv = []
    r = 0
    for c in range(L):
        if r == k:
            break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        rows = np.nonzero(M[:, c])[0]
        rows = rows[rows != r]
        if len(rows):
            M[rows] ^= M[r]
        piv.append(c)
        r += 1
    return M[:r], piv

def nullspace(M):
    """Basis (rows) of {v : M v = 0}."""
    k, n = M.shape
    R, piv = rref(M)
    free = [c for c in range(n) if c not in set(piv)]
    out = np.zeros((len(free), n), np.uint8)
    for t, f in enumerate(free):
        out[t, f] = 1
        for i, p in enumerate(piv):
            if R[i, f]:
                out[t, p] = 1
    return out

def rank(M):
    return len(rref(M)[1])

class Echelon:
    """Fully reduced echelon basis of a growing subspace of F_2^L."""
    def __init__(self, L):
        self.L = L
        self.E = np.zeros((0, L), np.uint8)
        self.piv = []

    def dim(self):
        return len(self.piv)

    def reduce(self, V):
        if not self.piv:
            return V.copy()
        return V ^ mm(V[:, self.piv], self.E)

    def add(self, V):
        """Add the rows of V; return the new echelon rows (a basis of the added part)."""
        V = self.reduce(np.atleast_2d(V))
        R, p = rref(V)
        if not p:
            return R
        if self.piv:
            self.E ^= mm(self.E[:, p], R)
        self.E = np.vstack([self.E, R])
        self.piv = self.piv + p
        order = np.argsort(self.piv)
        self.E = self.E[order]
        self.piv = [self.piv[i] for i in order]
        return R

    def coords(self, V):
        """Coordinates in the echelon basis of vectors known to lie in the span."""
        return V[:, self.piv]
