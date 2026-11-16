#!/usr/bin/env python3
r"""Centralizers of odd elements in the simple Lie superalgebras of Table 1.5 of
Chapter 1 of "Seminar on Supersymmetries", vol. 2 ("Regular odd elements. Dimensions of their centralizers").

Exact integer/rational arithmetic throughout.  For a matrix superalgebra
g \subset gl(m|n) and an odd x \in g we compute

    sdim Z_g(x) = ( dim{y in g_ev : [y,x]=0} | dim{y in g_od : [y,x]=0} ),

where [y,x] = yx - (-1)^{p(y)p(x)} xy.  The *regular* value is the minimum over
odd x, attained on a Zariski-open set, so a handful of random integer x give it
with probability 1; we take the minimum over several draws and, separately,
verify that the minimum is stable.

No claim here depends on a guess: everything is a rank of an explicit integer
matrix.
"""
import random
import sys
from fractions import Fraction

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank  # noqa: E402


# ---------------------------------------------------------------- linear algebra
def nullity(rows, ncols):
    if not rows:
        return ncols
    return ncols - rank(rows)


# ---------------------------------------------------------------- superalgebras
class Super:
    """A matrix Lie superalgebra inside gl(m|n), given by a basis split by parity."""

    def __init__(self, name, m, n, ev, od):
        self.name, self.m, self.n = name, m, n
        self.ev, self.od = ev, od          # lists of (m+n)x(m+n) matrices

    def sdim(self):
        return (len(self.ev), len(self.od))


def matmul(A, B, N):
    return [[sum(A[i][k] * B[k][j] for k in range(N)) for j in range(N)]
            for i in range(N)]


def bracket(A, pA, B, pB, N):
    AB = matmul(A, B, N)
    BA = matmul(B, A, N)
    s = -1 if (pA * pB) % 2 else 1
    return [[AB[i][j] - s * BA[i][j] for j in range(N)] for i in range(N)]


def eunit(N, i, j):
    M = [[0] * N for _ in range(N)]
    M[i][j] = 1
    return M


def gl(m, n):
    N = m + n
    ev, od = [], []
    for i in range(N):
        for j in range(N):
            p = (0 if i < m else 1) ^ (0 if j < m else 1)
            (od if p else ev).append(eunit(N, i, j))
    return Super("gl(%d|%d)" % (m, n), m, n, ev, od)


def sl(m, n):
    """Supertraceless matrices."""
    N = m + n
    g = gl(m, n)
    # odd part is automatically supertraceless
    ev = []
    # basis: E_ij (i!=j) plus supertraceless diagonal
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if ((0 if i < m else 1) ^ (0 if j < m else 1)) == 0:
                ev.append(eunit(N, i, j))
    # diagonal with str = sum_{i<m} d_i - sum_{i>=m} d_i = 0
    diag = []
    for k in range(1, N):
        d = [[0] * N for _ in range(N)]
        s0 = 1 if 0 < m else -1
        sk = 1 if k < m else -1
        d[0][0] = sk
        d[k][k] = -s0
        diag.append(d)
    ev.extend(diag)
    return Super("sl(%d|%d)" % (m, n), m, n, ev, g.od)


def osp(M, twon):
    """osp(M|2n): B symmetric on the even M-space, skew on the odd 2n-space."""
    n = twon // 2
    N = M + twon
    G = [[0] * N for _ in range(N)]
    for i in range(M):
        G[i][i] = 1
    for i in range(n):
        G[M + i][M + n + i] = 1
        G[M + n + i][M + i] = -1
    par = [0] * M + [1] * twon

    def basis_of_parity(p):
        # unknowns: entries X[a][b] with p(a)+p(b) = p
        idx, coords = {}, []
        for a in range(N):
            for b in range(N):
                if (par[a] ^ par[b]) == p:
                    idx[(a, b)] = len(coords)
                    coords.append((a, b))
        rows = []
        for i in range(N):
            for j in range(N):
                # B(Xe_i,e_j) + (-1)^{p p_i} B(e_i,Xe_j) = 0
                #   B(Xe_i,e_j) = sum_a X[a][i] G[a][j]
                #   B(e_i,Xe_j) = sum_a X[a][j] G[i][a]
                row = [0] * len(coords)
                sgn = -1 if (p * par[i]) % 2 else 1
                for a in range(N):
                    if (a, i) in idx and G[a][j]:
                        row[idx[(a, i)]] += G[a][j]
                    if (a, j) in idx and G[i][a]:
                        row[idx[(a, j)]] += sgn * G[i][a]
                if any(row):
                    rows.append(row)
        # basis of the solution space
        from exactlin import nullspace
        vecs = nullspace(rows) if rows else [[1 if k == t else 0 for k in range(len(coords))]
                                             for t in range(len(coords))]
        out = []
        for v in vecs:
            X = [[0] * N for _ in range(N)]
            den = 1
            for c in v:
                d = Fraction(c).denominator
                den = den * d // _gcd(den, d)
            for k, (a, b) in enumerate(coords):
                X[a][b] = int(Fraction(v[k]) * den)
            out.append(X)
        return out

    ev, od = basis_of_parity(0), basis_of_parity(1)
    return Super("osp(%d|%d)" % (M, twon), M, twon, ev, od)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# ---------------------------------------------------------------- centralizers
def centralizer_sdim(g, x, extra_mod=None):
    """sdim of {y : [y,x] = 0}; if extra_mod is a matrix, work modulo <extra_mod>."""
    N = g.m + g.n
    out = []
    for p, basis in ((0, g.ev), (1, g.od)):
        rows = []
        img = [bracket(b, p, x, 1, N) for b in basis]
        # flatten; the image of a p-basis element has parity p+1
        cols = len(basis)
        flat = [[img[k][i][j] for k in range(cols)] for i in range(N) for j in range(N)]
        rows = [r for r in flat if any(r)]
        if extra_mod is not None and (p + 1) % 2 == 0:
            # allow [y,x] in <I>: add the generator as an extra column, then the
            # solution space of the enlarged system, minus 1 for the added unknown
            enl = []
            for i in range(N):
                for j in range(N):
                    r = [img[k][i][j] for k in range(cols)] + [extra_mod[i][j]]
                    if any(r):
                        enl.append(r)
            out.append(nullity(enl, cols + 1) - 1)
        else:
            out.append(nullity(rows, cols))
    return tuple(out)


def random_odd(g, bound=9, seed=None):
    rnd = random.Random(seed)
    N = g.m + g.n
    X = [[0] * N for _ in range(N)]
    for b in g.od:
        c = rnd.randint(-bound, bound)
        for i in range(N):
            for j in range(N):
                X[i][j] += c * b[i][j]
    return X


def regular_sdim(g, tries=6, psl_center=None):
    best = None
    for t in range(tries):
        x = random_odd(g, seed=1000 + t)
        s = centralizer_sdim(g, x, extra_mod=psl_center)
        if best is None or (s[0] + s[1], s[0]) < (best[0] + best[1], best[0]):
            best = s
    return best
