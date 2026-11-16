"""Char-2 orthosymplectic candidates of sdim (9|8): V = (4|2), even form B = diag(B0, B1),
X in the algebra iff B X is symmetric, and B0 X_00 alternate (zero diagonal)."""
import itertools
from msa import nullspace

def mat(n, f): return [[f(i, j) for j in range(n)] for i in range(n)]
I = lambda n: mat(n, lambda i, j: int(i == j))
def Pi(n):  # hyperbolic: [[0,1],[1,0]] blocks
    return mat(n, lambda i, j: int(j == i ^ 1))

def blockdiag(A, B):
    a, b = len(A), len(B)
    return mat(a + b, lambda i, j: (A[i][j] if i < a and j < a else (B[i - a][j - a] if i >= a and j >= a else 0)))

def mm2(X, Y):
    n = len(X); return [[sum(X[i][k] & Y[k][j] for k in range(n)) & 1 for j in range(n)] for i in range(n)]

def candidate(B0, B1, alt0=True, alt1=False):
    B = blockdiag(B0, B1); n = 6
    vars_ = [(i, j) for i in range(n) for j in range(n)]
    def cond_rows(parity):
        rows = []
        allowed = [(i, j) for (i, j) in vars_ if ((i < 4) != (j < 4)) == bool(parity)]
        idx = {v: k for k, v in enumerate(allowed)}
        # (BX)_{ij} = sum_k B_ik X_kj ; symmetric: (BX)_ij + (BX)_ji = 0
        def lin(i, j):
            r = [0] * len(allowed)
            for k in range(n):
                if B[i][k] and (k, j) in idx: r[idx[(k, j)]] ^= 1
            return r
        for i in range(n):
            for j in range(i + 1, n):
                a, b = lin(i, j), lin(j, i); rows.append([x ^ y for x, y in zip(a, b)])
            if (alt0 and i < 4) or (alt1 and i >= 4):
                rows.append(lin(i, i))
        return allowed, rows
    out = []
    for parity in (0, 1):
        allowed, rows = cond_rows(parity)
        from gf2k import GFField
        F = GFField(2)
        sols = nullspace(F, rows, len(allowed))
        mats = []
        for s in sols:
            X = [[0] * n for _ in range(n)]
            for k, (i, j) in enumerate(allowed):
                X[i][j] = s[k] & 1
            mats.append(X)
        out.append(mats)
    return out
