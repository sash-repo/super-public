"""Census of the matrices up to PERMUTATION alone (a renumbering of the indices).

`vinb.py` counts up to S_n x {+-1}^n; this module counts up to the renumbering of the
indices only, the sign changes not divided out: this is the count of Statement 8.4.1.
Usage:  python permcensus.py 7

The canonical form is the lexicographic minimum of the signed strict lower triangle
over the permutations compatible with a Weisfeiler-Leman refinement of the matrix
(the refinement uses the signed entries, permutations being the only group acting).

Completeness of the level-by-level construction: every positive definite matrix of
size n has a positive definite principal submatrix of size n-1 (delete the last index),
and every index can be brought last by a permutation, so every orbit at level n is
reached from some orbit at level n-1.  Positive definiteness and having no odd cycle
are both inherited by principal submatrices, so the pruning loses nothing.
"""

import sys
from itertools import product, permutations

from vinb import (parse, is_posdef, is_connected, is_bipartite, det_int,
                  adjugate, extend)


def refine_signed(A):
    """WL colours using the signed entries; permutations preserve them."""
    n = len(A)
    color = [tuple(sorted(A[i][j] for j in range(n) if j != i)) for i in range(n)]
    color = _renumber(color)
    while True:
        sig = [(color[i], tuple(sorted((A[i][j], color[j])
                                       for j in range(n) if j != i))) for i in range(n)]
        new = _renumber(sig)
        if new == color:
            return color
        color = new


def _renumber(sig):
    order = {s: k for k, s in enumerate(sorted(set(sig)))}
    return [order[s] for s in sig]


def perm_canon(A):
    """Least signed lower triangle over all permutations respecting the WL colours."""
    n = len(A)
    color = refine_signed(A)
    classes = {}
    for i, c in enumerate(color):
        classes.setdefault(c, []).append(i)
    keys = sorted(classes)
    best = None
    for choice in product(*(permutations(classes[k]) for k in keys)):
        perm = [v for blk in choice for v in blk]
        s = tuple(A[perm[i]][perm[j]] for i in range(1, n) for j in range(i))
        if best is None or s < best:
            best = s
    return best


def from_triangle_tuple(t, n):
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2
    k = 0
    for i in range(1, n):
        for j in range(i):
            A[i][j] = A[j][i] = t[k]
            k += 1
    return tuple(tuple(r) for r in A)


def census(nmax, verbose=True):
    """Representatives up to permutation, level by level."""
    reps = {1: [((2,),)]}
    counts = {}
    for n in range(2, nmax + 1):
        seen = {}
        for A in reps[n - 1]:
            for B in extend(A):
                c = perm_canon(B)
                if c not in seen:
                    seen[c] = B
        reps[n] = list(seen.values())
        allc = len(reps[n])
        conn = sum(1 for B in reps[n] if is_connected(B))
        bip = sum(1 for B in reps[n] if is_bipartite(B))
        cbip = sum(1 for B in reps[n] if is_connected(B) and is_bipartite(B))
        counts[n] = (allc, conn, bip, cbip)
        if verbose:
            print(f'n={n:2d}  all={allc:8d}  connected={conn:8d} '
                  f' no-odd-cycle={bip:7d}  connected & no-odd-cycle={cbip:7d}', flush=True)
    return reps, counts


if __name__ == '__main__':
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    census(nmax)
