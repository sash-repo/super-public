"""Cross-check of the H^2 computation of nongraded.py by the second route.

Caranti-Mattarei compute H^2(L;F) not from the cochain complex but from
Der(L): with a nondegenerate associative form lambda on L,

    H^2(L;F) ~ SkDer(L)/ad(L)      in odd characteristic,
    H^2(L;F) ~ AltDer(L)/ad(L)     in characteristic 2 (their Remark 4.5),

where SkDer = {D : lambda(Dx,y) + lambda(x,Dy) = 0} and AltDer is the
subspace with lambda(Dx,x) = 0 as well (a strictly stronger condition at p=2).

This script computes lambda, verifies it is associative and nondegenerate,
and compares dim SkDer/ad and dim AltDer/ad with the dim H^2 that the
cochain complex gives.  The two routes are logically independent, so an
agreement pins the answer down.
"""
import sys
from itertools import combinations

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank, nullspace                          # noqa: E402
from nongraded import Ham                                     # noqa: E402


def lam_matrix(L):
    """lambda(x^(i)y^(j), x^(k)y^(l)) = (-1)^(i+j) [i+k = X-1][j+l = Y-1],
    plus lambda(e,e) = 1 for the omega_2 algebra (Blo58, Theorem 7)."""
    d, p = L.dim, L.p
    M = [[0] * d for _ in range(d)]
    for a, (i, j) in enumerate(L.basis):
        for b, (k, l) in enumerate(L.basis):
            if i + k == L.X - 1 and j + l == L.Y - 1:
                M[a][b] = ((-1) ** (i + j)) % p
    if L.form == 2:
        t = L.index[L.top]
        M[t][t] = (M[t][t] + 1) % p
    return M


def is_associative(L, M):
    T, p, d = L.table(), L.p, L.dim
    for a in range(d):
        for b in range(d):
            for c in range(d):
                lhs = sum(v * M[k][c] for k, v in T[a][b].items()) % p
                rhs = sum(v * M[a][k] for k, v in T[b][c].items()) % p
                if lhs != rhs:
                    return False, (a, b, c)
    return True, None


def der_basis(L):
    L.table()
    rows = []
    for a, b in combinations(range(L.dim), 2):
        rows.extend(L._der_rows(a, b))
    return nullspace(rows, p=L.p)


def constrain(L, M, basis, diagonal_only=False):
    """rows expressing 'D is skew' (or the extra alternating equations)
    in the coordinates of the given basis of Der(L)."""
    d, p = L.dim, L.p
    rows = []
    pairs = ([(x, x) for x in range(d)] if diagonal_only
             else [(x, y) for x in range(d) for y in range(x, d)])
    for (x, y) in pairs:
        row = []
        for D in basis:                       # D[m*d + t] = coeff of e_t in D e_m
            v = 0
            for t in range(d):
                v += D[x * d + t] * M[t][y] + D[y * d + t] * M[t][x]
            row.append(v % p)
        rows.append(row)
    return rows


def report(p, n1, n2, form):
    L = Ham(p, n1, n2, form)
    L.table()
    M = lam_matrix(L)
    ok, where = is_associative(L, M)
    nondeg = rank(M, p=p) == L.dim
    basis = der_basis(L)
    inner = L.inner_dim()

    sk = constrain(L, M, basis)
    dim_sk = len(basis) - rank(sk, p=p)
    # alternating = skew + the diagonal equations lambda(Dx,x) = 0
    dg = [[sum(D[x * L.dim + t] * M[t][x] for t in range(L.dim)) % p for D in basis]
          for x in range(L.dim)]
    dim_alt = len(basis) - rank(sk + dg, p=p)

    h2, z2, b2 = L.h2_dim()
    print("p=%d  H(2:(%d,%d);om_%d)  dim=%d" % (p, n1, n2, form, L.dim))
    print("  lambda: associative %s, nondegenerate %s" % (ok, nondeg))
    print("  dim Der=%d  ad=%d  SkDer=%d  AltDer=%d" % (len(basis), inner, dim_sk, dim_alt))
    print("  SkDer/ad = %d, AltDer/ad = %d   |   H^2 from cochains = %d"
          % (dim_sk - inner, dim_alt - inner, h2))
    print()


if __name__ == "__main__":
    for case in [(3, 1, 1, 2), (3, 1, 1, 0), (5, 1, 1, 0), (5, 1, 1, 2),
                 (2, 1, 1, 2), (2, 1, 2, 2), (2, 2, 2, 2), (2, 2, 2, 0),
                 (3, 1, 2, 2)]:
        report(*case)
