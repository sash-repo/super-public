#!/usr/bin/env python3
"""
A rigorous lower bound for the codimension of a proper subalgebra of a Lie
algebra L over F_2, and an exhaustive search for the maximal subalgebras of
small codimension.

Bound.  Every proper subspace lies in a hyperplane H = ker f.  For a fixed
non-zero f in L^* put
        B_f(x, y) := f([x, y]).
This form is bilinear, symmetric and alternating (B_f(x,x) = f([x,x]) = 0).  A
subalgebra M contained in H satisfies [M, M] subset M subset H, i.e. M is
totally isotropic for B_f.  A totally isotropic subspace of an alternating form
of rank 2r on a space of dimension m has dimension at most m - r.  Hence
        codim M >= 1 + min_f rank(B_f restricted to ker f)/2.

Exhaustive search.  A subalgebra of codimension c is the intersection of c
hyperplanes; we enumerate the c-dimensional subspaces U of L^* with U^perp
closed, for c = 1, 2, 3, ..., pruning with the bound above: a candidate f can
only take part if the isotropy bound allows codimension c.
"""

import glob
import itertools
import os
import sys

from css import LieF2, rref, in_span, all_elements
from thin import algebra_from_thin_table, load_table


def parity(x):
    return bin(x).count("1") & 1


def form(L, f):
    """the matrix (rows) of B_f(e_i, e_j) = f([e_i, e_j])"""
    n = L.n
    return [sum(parity(L.tab[i][j] & f) << j for j in range(n)) for i in range(n)]


def kernel(rows, n):
    piv = {}
    for r in rows:
        v = r
        for p in sorted(piv, reverse=True):
            if (v >> p) & 1:
                v ^= piv[p]
        if v:
            piv[v.bit_length() - 1] = v
    basis = []
    for i in range(n):
        if i in piv:
            continue
        v = 1 << i
        for p in sorted(piv):
            if parity(piv[p] & v):
                v ^= 1 << p
        basis.append(v)
    for v in basis:
        for r in rows:
            assert not parity(r & v)
    return rref(basis)


def restrict_rank(L, f):
    """rank of B_f restricted to ker f"""
    n = L.n
    B = form(L, f)
    K = kernel([f], n)
    rows = []
    for u in K:
        r = 0
        for j, v in enumerate(K):
            s = 0
            for i in range(n):
                if (u >> i) & 1 and parity(B[i] & v):
                    s ^= 1
            if s:
                r |= 1 << j
        rows.append(r)
    return len(rref(rows))


def isotropy_bound(L):
    n = L.n
    best = None
    for f in range(1, 1 << n):
        r = restrict_rank(L, f)
        c = 1 + (r + 1) // 2
        if best is None or c < best[0]:
            best = (c, f, r)
    return best


def subalgebras_of_codim(L, c, verbose=False):
    """all subalgebras of codimension exactly c, by enumerating c-dimensional
    subspaces of L^* in echelon form."""
    n = L.n
    out = []
    # enumerate c-subspaces of the dual by their reduced echelon bases
    for pivots in itertools.combinations(range(n), c):
        free = [i for i in range(n) if i not in pivots]
        nfree = len(free)
        for assign in range(1 << (c * nfree)):
            U = []
            a = assign
            for k, p in enumerate(pivots):
                v = 1 << p
                for j in free:
                    if j < p:
                        continue
                    if (a >> 0) & 1:
                        v |= 1 << j
                    a >>= 1
                U.append(v)
            M = kernel(U, n)
            if len(M) != n - c:
                continue
            if L.is_subalgebra(M):
                out.append(M)
    return out


def main(paths):
    for path in sorted(paths):
        name = os.path.basename(path)[:-4]
        L = algebra_from_thin_table(load_table(path), name)
        c, f, r = isotropy_bound(L)
        print("%-14s dim %2d :  every proper subalgebra has codimension >= %d"
              "   (worst hyperplane: rank of B_f on ker f is %d)"
              % (name, L.n, c, r))


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("css-bases/thin15/*.txt")))
