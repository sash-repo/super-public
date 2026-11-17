#!/usr/bin/env python3
"""Matrix models of the (super)algebras of arXiv:1906.01380, built from scratch.

Every algebra is produced as a basis of matrices over Q together with the parity
of each basis element; nothing is asserted, the defining linear system is solved
and the result is checked (closure under the bracket, dimension).
"""

from fractions import Fraction
from sant import mmul, madd, flat


# ------------------------------------------------------------------ linear algebra

def nullspace(rows, ncols):
    """Basis of the kernel of the matrix with these rows, exactly over Q."""
    M = [list(map(Fraction, r)) for r in rows]
    piv = {}
    r = 0
    for c in range(ncols):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pr = M[r]
        M[r] = [x / pr[c] for x in pr]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv[c] = r
        r += 1
        if r == len(M):
            break
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for fc in free:
        v = [Fraction(0)] * ncols
        v[fc] = Fraction(1)
        for c, row in piv.items():
            v[c] = -M[row][fc]
        basis.append(v)
    return basis


def parity_vec(m, n):
    return [0] * m + [1] * n


# ------------------------------------------------------------------ the algebras

def _mat_from_vec(v, N):
    return [[v[i * N + j] for j in range(N)] for i in range(N)]


def _homog_basis(N, pv, cond):
    """Basis of {X homogeneous : cond(X_as_vector, p(X)) = 0}, split by parity."""
    out = []
    for pX in (0, 1):
        # coordinates allowed for this parity
        idx = [(i, j) for i in range(N) for j in range(N) if (pv[i] + pv[j]) % 2 == pX]
        rows = cond(idx, pX, N, pv)
        ker = nullspace(rows, len(idx)) if rows else \
            [[Fraction(1) if t == s else Fraction(0) for t in range(len(idx))]
             for s in range(len(idx))]
        for v in ker:
            M = [[Fraction(0)] * N for _ in range(N)]
            for (c, (i, j)) in enumerate(idx):
                M[i][j] = v[c]
            out.append((M, pX))
    return out


def gl(m, n):
    N = m + n
    pv = parity_vec(m, n)
    return _homog_basis(N, pv, lambda idx, pX, N, pv: []), pv


def sl(m, n):
    N = m + n
    pv = parity_vec(m, n)

    def cond(idx, pX, N, pv):
        if pX == 1:
            return []
        row = [0] * len(idx)
        for c, (i, j) in enumerate(idx):
            if i == j:
                row[c] = 1 if pv[i] == 0 else -1
        return [row]
    return _homog_basis(N, pv, cond), pv


def aut_form(G, pv, pG):
    """{X : (X^T G)_{ij} + (-1)^{p(X) p(i)} (G X)_{ij} = 0 for all i,j}."""
    N = len(pv)

    def cond(idx, pX, N, pv):
        rows = []
        for i in range(N):
            for j in range(N):
                row = [0] * len(idx)
                for c, (a, b) in enumerate(idx):
                    # (X^T G)_{ij} = sum_a X_{a i} G_{a j}
                    if b == i:
                        row[c] += G[a][j]
                    # (G X)_{ij} = sum_b G_{i b} X_{b j}
                    if b == j and G[i][a] != 0:
                        row[c] += (1 if (pX * pv[i]) % 2 == 0 else -1) * G[i][a]
                if any(row):
                    rows.append(row)
        return rows
    return _homog_basis(N, pv, cond)


def osp(m, n2):
    """osp(m|2n): Gram diag(1_m, J_{2n}), J = antidiag(1_n,-1_n).  n2 = 2n."""
    n = n2 // 2
    N = m + n2
    pv = parity_vec(m, n2)
    G = [[Fraction(0)] * N for _ in range(N)]
    for i in range(m):
        G[i][i] = Fraction(1)
    for i in range(n):
        G[m + i][m + n + i] = Fraction(1)
        G[m + n + i][m + i] = Fraction(-1)
    return aut_form(G, pv, 0), pv


def pe(n, anti=False):
    """pe(n): odd non-degenerate form on n|n, Gram [[0,1],[1,0]] (or [[0,1],[-1,0]])."""
    N = 2 * n
    pv = parity_vec(n, n)
    G = [[Fraction(0)] * N for _ in range(N)]
    for i in range(n):
        G[i][n + i] = Fraction(1)
        G[n + i][i] = Fraction(-1) if anti else Fraction(1)
    return aut_form(G, pv, 1), pv


def spe(n, anti=False):
    b, pv = pe(n, anti)
    N = 2 * n
    out = []
    for pX in (0, 1):
        blk = [M for M, p in b if p == pX]
        if not blk:
            continue
        if pX == 0:
            # impose str = 0
            rows = [[sum(M[i][i] for i in range(n)) - sum(M[i][i] for i in range(n, N))
                     for M in blk]]
            ker = nullspace(rows, len(blk))
            for v in ker:
                M = [[sum(v[t] * blk[t][i][j] for t in range(len(blk)))
                      for j in range(N)] for i in range(N)]
                out.append((M, 0))
        else:
            out.extend((M, 1) for M in blk)
    return out, pv


def q(n):
    N = 2 * n
    pv = parity_vec(n, n)
    out = []
    for i in range(n):
        for j in range(n):
            M = [[Fraction(0)] * N for _ in range(N)]
            M[i][j] = M[n + i][n + j] = Fraction(1)
            out.append((M, 0))
            M = [[Fraction(0)] * N for _ in range(N)]
            M[i][n + j] = M[n + i][j] = Fraction(1)
            out.append((M, 1))
    return out, pv


def sq(n):
    b, pv = q(n)
    N = 2 * n
    out = []
    for M, p in b:
        if p == 0:
            out.append((M, p))
    odd = [M for M, p in b if p == 1]
    rows = [[sum(M[n + i][i] for i in range(n)) for M in odd]]   # qtr = tr B
    for v in nullspace(rows, len(odd)):
        M = [[sum(v[t] * odd[t][i][j] for t in range(len(odd)))
              for j in range(N)] for i in range(N)]
        out.append((M, 1))
    return out, pv


# ------------------------------------------------------- purely even classical ones

def o(N):
    pv = [0] * N
    G = [[Fraction(1 if i == j else 0) for j in range(N)] for i in range(N)]
    return aut_form(G, pv, 0), pv


def sp(N):
    n = N // 2
    pv = [0] * N
    G = [[Fraction(0)] * N for _ in range(N)]
    for i in range(n):
        G[i][n + i] = Fraction(1)
        G[n + i][i] = Fraction(-1)
    return aut_form(G, pv, 0), pv


FANO = [(i, (i + 1) % 7, (i + 3) % 7) for i in range(7)]


def g2():
    """g2 = {A in o(7) : A annihilates the octonionic 3-form}."""
    base, pv = o(7)
    mats = [M for M, p in base]
    phi = {}
    for (a, b, c) in FANO:
        for perm, s in (((a, b, c), 1), ((b, c, a), 1), ((c, a, b), 1),
                        ((b, a, c), -1), ((a, c, b), -1), ((c, b, a), -1)):
            phi[perm] = s
    rows = []
    for i in range(7):
        for j in range(7):
            for k in range(7):
                row = []
                for M in mats:
                    s = Fraction(0)
                    for mm in range(7):
                        s -= M[mm][i] * phi.get((mm, j, k), 0)
                        s -= M[mm][j] * phi.get((i, mm, k), 0)
                        s -= M[mm][k] * phi.get((i, j, mm), 0)
                    row.append(s)
                if any(row):
                    rows.append(row)
    ker = nullspace(rows, len(mats))
    out = []
    for v in ker:
        M = [[sum(v[t] * mats[t][i][j] for t in range(len(mats)))
              for j in range(7)] for i in range(7)]
        out.append((M, 0))
    return out, pv


# ------------------------------------------------------------------ sanity checks

def bracket(X, Y, pX, pY):
    XY = mmul(X, Y)
    YX = mmul(Y, X)
    s = -1 if (pX * pY) % 2 == 1 else 1
    return [[XY[i][j] - s * YX[i][j] for j in range(len(X))] for i in range(len(X))]


def check_closed(basis, name):
    from sant import in_span
    vecs = [flat(M) for M, p in basis]
    for a, (X, pX) in enumerate(basis):
        for b, (Y, pY) in enumerate(basis):
            if not in_span(flat(bracket(X, Y, pX, pY)), vecs):
                return f"{name}: NOT closed under the bracket ({a},{b})"
    return f"{name}: closed, sdim = ({sum(1 for _, p in basis if p == 0)}|" \
           f"{sum(1 for _, p in basis if p == 1)})"


if __name__ == "__main__":
    for name, (b, pv) in [("sl(2|1)", sl(2, 1)), ("osp(3|2)", osp(3, 2)),
                          ("osp(1|4)", osp(1, 4)), ("pe(2)", pe(2)),
                          ("pe(3)", pe(3)), ("spe(3)", spe(3)),
                          ("q(2)", q(2)), ("sq(2)", sq(2)),
                          ("o(5)", o(5)), ("o(6)", o(6)), ("sp(4)", sp(4)),
                          ("g2", g2())]:
        print(check_closed(b, name))
