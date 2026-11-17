#!/usr/bin/env python3
"""The octonions, the Albert algebra, and from them f4 (26-dimensional module)
and the operators from which runexc.py builds e6 (27-dimensional module).

    f4 = Der(J),  J the 27-dimensional exceptional Jordan algebra, acting on the
         traceless part J_0;
    e6 = Der(J) + {L_X : tr X = 0},  L_X the Jordan multiplication by X.

Nothing is asserted: Der(J) is the kernel of the defining linear system, solved
exactly over Q, and its dimension (52) is printed.
"""

import sys
from fractions import Fraction

# exactlin.py (exact linear algebra) is imported from this folder

# ------------------------------------------------------------------ octonions
FANO = [((i % 7) + 1, ((i + 1) % 7) + 1, ((i + 3) % 7) + 1) for i in range(7)]

MUL = {}
for a in range(8):
    MUL[(0, a)] = (a, 1)
    MUL[(a, 0)] = (a, 1)
for i in range(1, 8):
    MUL[(i, i)] = (0, -1)
for (a, b, c) in FANO:
    for (x, y, z) in ((a, b, c), (b, c, a), (c, a, b)):
        MUL[(x, y)] = (z, 1)
        MUL[(y, x)] = (z, -1)


def omul(u, v):
    """Product of two octonions given as length-8 coefficient lists."""
    w = [0] * 8
    for i, a in enumerate(u):
        if a == 0:
            continue
        for j, b in enumerate(v):
            if b == 0:
                continue
            k, s = MUL[(i, j)]
            w[k] += s * a * b
    return w


def oconj(u):
    return [u[0]] + [-x for x in u[1:]]


def check_octonions():
    for i in range(8):
        for j in range(8):
            u = [1 if t == i else 0 for t in range(8)]
            v = [1 if t == j else 0 for t in range(8)]
            uv = omul(u, v)
            # ||uv|| = ||u|| ||v||
            if sum(x * x for x in uv) != 1:
                return "norm fails"
    # alternativity, a sample
    for (i, j) in [(1, 2), (2, 4), (3, 5), (1, 6)]:
        u = [1 if t == i else 0 for t in range(8)]
        v = [1 if t == j else 0 for t in range(8)]
        if omul(omul(u, u), v) != omul(u, omul(u, v)):
            return "alternativity fails"
    return "octonions: composition and alternativity OK"


# --------------------------------------------------------- the Albert algebra
# X = [[x1, c3, conj(c2)], [conj(c3), x2, c1], [c2, conj(c1), x3]]

def jmat(diag, offs):
    c1, c2, c3 = offs
    return [[[diag[0]] + [0] * 7, c3, oconj(c2)],
            [oconj(c3), [diag[1]] + [0] * 7, c1],
            [c2, oconj(c1), [diag[2]] + [0] * 7]]


def jbasis():
    b = []
    for d in range(3):
        b.append(jmat([1 if t == d else 0 for t in range(3)],
                      [[0] * 8, [0] * 8, [0] * 8]))
    for k in range(3):
        for a in range(8):
            offs = [[0] * 8, [0] * 8, [0] * 8]
            offs[k] = [1 if t == a else 0 for t in range(8)]
            b.append(jmat([0, 0, 0], offs))
    return b


def mmul3(A, B):
    return [[[sum(x) for x in zip(*[omul(A[i][t], B[t][j]) for t in range(3)])]
             for j in range(3)] for i in range(3)]


def jprod(A, B):
    X = mmul3(A, B)
    Y = mmul3(B, A)
    return [[[Fraction(x + y, 2) for x, y in zip(X[i][j], Y[i][j])]
             for j in range(3)] for i in range(3)]


def coords(A):
    """Coordinates of a Hermitian matrix in the basis jbasis()."""
    v = [A[0][0][0], A[1][1][0], A[2][2][0]]
    for k, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):
        v.extend(A[i][j])
    return v


def structure_constants():
    B = jbasis()
    n = len(B)
    C = {}
    for i in range(n):
        for j in range(i, n):
            v = coords(jprod(B[i], B[j]))
            C[(i, j)] = v
            C[(j, i)] = v
    return C, n


# ------------------------------------------------------------------ f4 and e6

def der_J():
    """f4 = Der(J) as 27x27 matrices, then restricted to the traceless J_0."""
    from exactlin import nullspace
    C, n = structure_constants()
    rows = []
    for i in range(n):
        for j in range(i, n):
            for m in range(n):
                row = [Fraction(0)] * (n * n)
                for t in range(n):
                    row[m * n + t] += C[(i, j)][t]
                    row[t * n + i] -= C[(t, j)][m]
                    row[t * n + j] -= C[(i, t)][m]
                if any(row):
                    rows.append(row)
    ker = nullspace(rows)
    return [[[v[m * n + t] for t in range(n)] for m in range(n)] for v in ker]


def L_operators():
    """Jordan multiplication operators L_X, X in J, as 27x27 matrices."""
    C, n = structure_constants()
    out = []
    for i in range(n):
        out.append([[C[(i, t)][m] for t in range(n)] for m in range(n)])
    return out


def trace_form_coeffs():
    """The linear form tr on J in the basis jbasis(): tr X = x1 + x2 + x3."""
    C, n = structure_constants()
    return [1, 1, 1] + [0] * (n - 3)


def restrict_to_J0(mats):
    """Rewrite operators that preserve tr = 0 in a basis of the 26-dimensional J_0."""
    n = 27
    tr = trace_form_coeffs()
    # basis of J_0: e_1 - e_2, e_2 - e_3, and the 24 off-diagonal ones
    B = []
    v = [Fraction(0)] * n
    v[0], v[1] = Fraction(1), Fraction(-1)
    B.append(v[:])
    v = [Fraction(0)] * n
    v[1], v[2] = Fraction(1), Fraction(-1)
    B.append(v[:])
    for t in range(3, n):
        v = [Fraction(0)] * n
        v[t] = Fraction(1)
        B.append(v[:])
    assert all(sum(a * b for a, b in zip(tr, v)) == 0 for v in B)
    from exactlin import solve
    out = []
    A = [[B[c][r] for c in range(len(B))] for r in range(n)]     # n x 26
    for M in mats:
        cols = []
        for v in B:
            w = [sum(M[r][c] * v[c] for c in range(n)) for r in range(n)]
            cols.append(solve(A, w))
        out.append([[cols[c][r] for c in range(len(B))] for r in range(len(B))])
    return out


if __name__ == "__main__":
    print(check_octonions())
    C, n = structure_constants()
    print(f"Albert algebra: dim = {n}, commutative:",
          all(C[(i, j)] == C[(j, i)] for i in range(n) for j in range(n)))
    f4 = der_J()
    print(f"dim Der(J) = {len(f4)}")
    L = L_operators()
    print(f"L operators: {len(L)}")
    f4_26 = restrict_to_J0(f4)
    print(f"f4 on J_0: {len(f4_26)} matrices of size {len(f4_26[0])}")

