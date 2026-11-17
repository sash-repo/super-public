"""The retractable non-split (1|3)-dimensional supermanifolds over CP^1 with
retract Lambda(K + K + K): how many there are (Statement A.5.5 of the text).

The class of such a supermanifold is an element of
    H^1(CP^1; N (x) Lambda^3 N^*) = H^1(O(-4)) (x) (N (x) Lambda^3 N^*)_fibre
                                  = V (x) W,     dim V = dim W = 3,
V being the 3-dimensional irreducible sl(2)-module (the exponents -1, -2, -3 of
x) and W the fibre, on which GL(3) = Aut E acts.  Two classes give isomorphic
supermanifolds iff they lie in one orbit of Aut of the split model, i.e., of
GL(W) x GO(V) -- the second factor being PGL(2) acting on V, which preserves
the invariant quadratic form Q of V up to a scalar.

An orbit of GL(W) x GO(V) on Hom(W^*, V) = V (x) W is given by the rank r and
by the O(Q)-type of the IMAGE (a subspace of V of dimension r, classified by
whether Q restricted to it is degenerate), so the census is

    r = 3:  the image is V                    1 class
    r = 2:  a plane, Q on it (non)degenerate  2 classes
    r = 1:  a line, Q(v) = 0 or not           2 classes
                                             ----------
                                              5 classes,

and this engine checks it: it computes sdim v(X) for a large sample and finds
that it is a function of that invariant alone, with five distinct values.
"""

from fractions import Fraction
from itertools import product
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import rank, nullspace                              # noqa: E402

import ns                                                          # noqa: E402
import sl2inv as S                                                 # noqa: E402
import quot                                                        # noqa: E402


KS = [2, 2, 2]
EXPS = (-1, -2, -3)


# ------------------------------------- the invariant quadratic form on V

def invariant_form():
    """The sl(2)-invariant quadratic form on the exponent space of
    H^1(N (x) Lambda^3 N^*), computed from the honest action of the global
    vector fields d/dx, x d/dx, x^2 d/dx on the cocycles."""
    Q = quot.Quot(KS, 2)
    # the three cocycles x^e xi_1 xi_2 xi_3 d/dxi_1 span a copy of V
    import bashkin as B
    basis = [B.w((1, 2, 3), 1, e) for e in EXPS]
    idx = [Q.coords(b) for b in basis]
    # the copy of sl(2) in v(split)_0: the naive x^2 d/dx is NOT a vector
    # field of the supermanifold, it has to be lifted by x * (the grading
    # operator), and it is that term which makes the action have trace zero.
    e = S.d_x(3, ns.g_from((), ns.lp((0, 1))))                       # d/dx
    f = S.fld_add(S.d_x(3, ns.g_from((), ns.lp((2, 1)))),            # x^2 d/dx
                  S._x_times(3, S.nabla(3, KS)))                     # + x nabla
    h = S.bracket_even(3, e, f)
    mats = {}
    for name, X in (("e", e), ("f", f), ("h", h)):
        cols = []
        for b in basis:
            Y = S.coc_to_field(3, b)
            Z = S.bracket_even(3, X, Y)
            c = Q.coords(S.field_to_coc(3, 2, Z))
            # express in the chosen 3-dimensional subspace
            cols.append(_in_span(idx, c))
        mats[name] = [[cols[j][i] for j in range(3)] for i in range(3)]
    # solve for the symmetric G with X^t G + G X = 0 for X = e, f, h
    unk = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    rows = []
    for name in ("e", "f", "h"):
        M = mats[name]
        for a in range(3):
            for b in range(a, 3):
                row = []
                for (p, q) in unk:
                    v = Fraction(0)
                    for k in range(3):
                        if (k, b) == (p, q) or (b, k) == (p, q):
                            v += M[k][a]
                        if (a, k) == (p, q) or (k, a) == (p, q):
                            v += M[k][b]
                    row.append(v)
                rows.append(row)
    sol = nullspace(rows)
    assert len(sol) == 1, f"expected a one-dimensional space of forms, got {len(sol)}"
    g = sol[0]
    G = [[Fraction(0)] * 3 for _ in range(3)]
    for (p, q), v in zip(unk, g):
        G[p][q] = v
        G[q][p] = v
    # normalize
    piv = next(v for row in G for v in row if v)
    return [[v / piv for v in row] for row in G], mats


def _in_span(idx, c):
    """Coordinates of c in the span of the rows idx (exact)."""
    from exactlin import solve
    mat = [[idx[j][i] for j in range(len(idx))] for i in range(len(idx[0]))]
    s = solve(mat, list(c))
    assert s is not None, "the bracket left the chosen 3-dimensional subspace"
    return s


# ------------------------------------------------------------ the census

def sm_of(M):
    z = {}
    for i, e in enumerate(EXPS):
        for t in (1, 2, 3):
            if M[i][t - 1]:
                z[t] = ns.g_add(z.get(t, ns.g_zero()),
                                ns.g_from((1, 2, 3), ns.lp((e, M[i][t - 1]))))
    return ns.SM.from_cocycle(KS, zx=None, zxi=z)


def invariant(M, G):
    """(rank, type of Q on the image in V: 'nondeg' / 'deg').

    A class M in V (x) W is a map W^* -> V; GL(W) moves the source freely, so
    the whole invariant is the image, an r-dimensional subspace of V, taken up
    to the orthogonal group of Q.  Subspaces of a 3-dimensional quadratic
    space are classified by the dimension and by whether Q restricted to them
    is degenerate."""
    r = rank([[Fraction(c) for c in row] for row in M])
    if r == 0:
        return (0, "-")
    if r == 3:
        return (3, "-")
    img = [[Fraction(M[i][t]) for i in range(3)] for t in range(3)]  # columns
    Gk = [[sum(u[a] * G[a][b] * v[b] for a in range(3) for b in range(3))
           for v in img] for u in img]
    deg = rank(Gk) < r
    return (r, "deg" if deg else "nondeg")


def main():
    G, mats = invariant_form()
    print("the sl(2)-invariant quadratic form on the exponent space "
          "(basis x^{-1}, x^{-2}, x^{-3}):")
    for row in G:
        print("   ", [str(v) for v in row])
    seen = {}
    n = 0
    # the sample: every matrix with entries in {-1, 0, 1, 2} in the first two
    # columns and a third column from {-1, 0, 1}, so that all five invariants
    # occur many times over
    sample = []
    for a in product((-1, 0, 1, 2), repeat=3):
        for b in product((-1, 0, 1), repeat=3):
            for c in product((-1, 0, 1), repeat=3):
                sample.append([[a[i], b[i], c[i]] for i in range(3)])
    for M in sample:
        inv = invariant(M, G)
        if inv[0] == 0:
            continue
        if inv in seen and len(seen[inv]) >= 12:
            continue
        s = sm_of(M).sdim_vect(degU=8)
        seen.setdefault(inv, []).append((s, M))
        n += 1
    print(f"\n{n} supermanifolds computed; sdim v by invariant:")
    bad = 0
    for inv in sorted(seen):
        ss = {s for s, _ in seen[inv]}
        print(f"   rank {inv[0]}, image {inv[1]:7s}: "
              f"sdim v in {sorted(ss)}   ({len(seen[inv])} samples)")
        if len(ss) != 1:
            bad += 1
            print("      ^ NOT constant on the orbit", seen[inv])
    allv = {next(iter({s for s, _ in seen[inv]})) for inv in seen}
    print(f"\n   {len(seen)} invariants, {len(allv)} distinct superdimensions: "
          f"{sorted(allv)}")
    ok = (bad == 0 and len(seen) == 5 and len(allv) == 5)
    print(f"\n   [{'PASS' if ok else 'FAIL'}] five classes, told apart by "
          f"sdim v, constant on each orbit")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
