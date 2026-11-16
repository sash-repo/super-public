#!/usr/bin/env python3
"""k(1|5): there is no family -- i(osp(5|2)) is rigid inside k(1|5).

Grade k(1|n) by deg t = 2, deg y_i = 1, so deg K_f = deg f - 2:

    g_{-2} = <K_1>, g_{-1} = <K_{y_i}>, g_0 = <K_t, K_{y_i y_j}> = C + o(n),
    g_1 = <K_{t y_i}> (+) <K_{y_i y_j y_k}> = V (+) Lambda^3 V.

For n = 4 one has Lambda^3 V isomorphic to V, whence the one-parameter family
osp_a(4|2) of k14family.py: the g_0-stable complements to Lambda^3 V in g_1 are
the graphs of a.sigma with sigma the (unique) equivariant map V --> Lambda^3 V.

This script computes dim Hom_{o(n)}(V, Lambda^3 V) directly, for n = 3,...,6,
by intertwining ad K_{y_i y_j}: it is 0 for n = 3, 5 and 1 for n = 4, and for
n = 6 the source of the phenomenon is different (Lambda^3 splits).  Hence for
n = 5 the graded subalgebra h with h_- = g_-, h_0 = g_0 and h_1 = V is unique,
i.e. i(osp(5|2)) sits in k(1|5) in exactly one way.

It also lists the o(n)-submodules of g_1 by computing the dimension of the
commutant of the o(n)-action, whose dimension is the number of irreducible
summands counted with multiplicity squared.
"""
from itertools import combinations
from fractions import Fraction as Fr
from sympy import Symbol, Matrix, linsolve, expand

from superN import Contact, clean, mono

GRAM = {
    3: ([[1, 0, 0], [0, 0, 1], [0, 1, 0]], ['th', 'xi', 'eta']),
    4: ([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        ['xi1', 'eta1', 'xi2', 'eta2']),
    5: ([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]],
        ['th', 'xi1', 'eta1', 'xi2', 'eta2']),
    6: ([[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]],
        ['xi1', 'eta1', 'xi2', 'eta2', 'xi3', 'eta3']),
}


def msk(idx):
    m = 0
    for i in idx:
        m |= 1 << i
    return m


def coords(u, basis):
    ks = [next(iter(b)) for b in basis]
    v = [u.get(k, 0) for k in ks]
    assert not clean({k: c for k, c in u.items() if k not in ks}), \
        "not in the span"
    return v


def report(n):
    B, names = GRAM[n]
    alg = Contact(n, B, names=names)
    bra = alg.kb
    TY = [mono(1, 1 << i) for i in range(n)]
    CUB = [mono(0, msk(c)) for c in combinations(range(n), 3)]
    QUAD = [mono(0, msk(c)) for c in combinations(range(n), 2)]
    p, q = len(TY), len(CUB)

    # the equivariant maps V --> Lambda^3 V
    M = [[Symbol(f'm{i}_{j}') for j in range(q)] for i in range(p)]
    eqs = []
    for Q in QUAD:
        # ad K_Q on V and on Lambda^3 V
        A = [coords(bra(Q, b), TY) for b in TY]          # A[i][k]: {Q,v_i} = sum A[i][k] v_k
        C = [coords(bra(Q, b), CUB) for b in CUB]
        # equivariance: sigma({Q,v_i}) = {Q, sigma(v_i)}
        for i in range(p):
            lhs = [sum(A[i][k] * M[k][j] for k in range(p)) for j in range(q)]
            rhs = [sum(M[i][k] * C[k][j] for k in range(q)) for j in range(q)]
            eqs += [expand(a - b) for a, b in zip(lhs, rhs)]
    vs = [M[i][j] for i in range(p) for j in range(q)]
    A = Matrix([[eq.coeff(v) for v in vs] for eq in eqs])
    dimhom = len(vs) - A.rank()
    print(f"n = {n}: dim V = {p}, dim Lambda^3 V = {q}, "
          f"dim Hom_o({n})(V, Lambda^3 V) = {dimhom}")

    # the commutant of the o(n)-action on g_1 = V + Lambda^3 V
    basis = TY + CUB
    d = p + q
    X = [[Symbol(f'x{i}_{j}') for j in range(d)] for i in range(d)]
    eqs = []
    for Q in QUAD:
        A = [coords(bra(Q, b), basis) for b in basis]
        for i in range(d):
            lhs = [sum(A[i][k] * X[k][j] for k in range(d)) for j in range(d)]
            rhs = [sum(X[i][k] * A[k][j] for k in range(d)) for j in range(d)]
            eqs += [expand(a - b) for a, b in zip(lhs, rhs)]
    vs = [X[i][j] for i in range(d) for j in range(d)]
    A = Matrix([[eq.coeff(v) for v in vs] for eq in eqs])
    print(f"        dim End_o({n})(g_1) = {len(vs) - A.rank()} "
          f"(2 = two non-isomorphic irreducibles, 4 = a family, "
          f"5 = Lambda^3 splits)")


if __name__ == '__main__':
    import sys
    for n in ([int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1
              else [3, 4, 5, 6]):
        report(n)
