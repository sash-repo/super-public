#!/usr/bin/env python3
"""Statement 3.1.2: which of the 370 Cartan matrices of Subsections 9.1 and 9.2 are
symmetrizable, and the signature of their symmetrizations.

For a symmetrizable Cartan matrix A = diag(d)^{-1} S with S symmetric and d > 0, the sign of
det A is (-1)^{n_-}, where n_- is the number of negative squares of S, formula (28).  We count
the symmetrizable matrices (the names beginning with H and S are expected to be exactly the
symmetrizable ones, the names beginning with NH and NS the others), compute n_- for each, and
check formula (28).  Exact rational arithmetic.

Usage:  python symm.py
"""
from fractions import Fraction

from tables import load, det

def symmetrizer(A):
    """Return the positive rational d with diag(d) A symmetric, or None.

    A is symmetrizable iff there are d_i != 0 with d_i A_ij = d_j A_ji for all i, j.
    We solve this on the graph whose edges are the pairs with A_ij != 0.
    """
    n = len(A)
    d = [None] * n
    for s in range(n):
        if d[s] is not None:
            continue
        d[s] = Fraction(1)
        stack = [s]
        while stack:
            i = stack.pop()
            for j in range(n):
                if i == j:
                    continue
                aij, aji = A[i][j], A[j][i]
                if aij == 0 and aji == 0:
                    continue
                if aij == 0 or aji == 0:
                    return None                      # d_i A_ij = d_j A_ji impossible
                dj = d[i] * aij / aji
                if d[j] is None:
                    d[j] = dj
                    stack.append(j)
                elif d[j] != dj:
                    return None
    return d


def inertia(S):
    """(n_+, n_0, n_-) of a symmetric rational matrix, by symmetric Gaussian sweep."""
    n = len(S)
    S = [row[:] for row in S]
    pos = neg = zero = 0
    idx = list(range(n))
    while idx:
        piv = next((i for i in idx if S[i][i] != 0), None)
        if piv is None:
            # every remaining diagonal entry is 0: find an off-diagonal one
            pair = next(((i, j) for i in idx for j in idx if i != j and S[i][j] != 0), None)
            if pair is None:
                zero += len(idx)
                break
            i, j = pair
            for k in idx:                            # row_i += row_j, col_i += col_j
                S[i][k] += S[j][k]
            for k in idx:
                S[k][i] += S[k][j]
            continue
        p = S[piv][piv]
        if p > 0:
            pos += 1
        else:
            neg += 1
        idx.remove(piv)
        for i in idx:
            if S[i][piv]:
                f = S[i][piv] / p
                for k in idx:
                    S[i][k] -= f * S[piv][k]
                S[i][piv] = Fraction(0)
        for i in idx:
            S[piv][i] = Fraction(0)
    return pos, zero, neg


def main():
    print("=" * 78)
    print("symmetrizability and signature of the 370 Cartan matrices of Subsections 9.1, 9.2")
    print("=" * 78)

    count = {}
    bad_sign, bad_name = [], []
    neg_counts = {}
    for e in load():
        A, name = e["A"], e["name"]
        sect = e["subsection"]
        d = symmetrizer(A)
        sym = d is not None and all(x > 0 for x in d)
        count[(sect, sym)] = count.get((sect, sym), 0) + 1
        if sym == name.startswith("N"):
            bad_name.append(name)
        if not sym:
            continue
        S = [[d[i] * A[i][j] for j in range(len(A))] for i in range(len(A))]
        npos, nzero, nneg = inertia(S)
        neg_counts[nneg] = neg_counts.get(nneg, 0) + 1
        if (det(A) > 0) != (nneg % 2 == 0):
            bad_sign.append((name, det(A), nneg))

    for sect, what in (("9.1", "almost affine Lie superalgebras"), ("9.2", "hyperbolic Lie algebras")):
        s, ns = count.get((sect, True), 0), count.get((sect, False), 0)
        print("Subsection %s (%s): %d matrices, %d symmetrizable, %d not" % (sect, what, s + ns, s, ns))
    print("total symmetrizable: %d, non-symmetrizable: %d"
          % (sum(v for (s, y), v in count.items() if y), sum(v for (s, y), v in count.items() if not y)))
    print("number n_- of negative squares of the symmetrized matrix:")
    for k in sorted(neg_counts):
        print("    n_- = %d :  %d matrices" % (k, neg_counts[k]))
    print("sign(det A) = (-1)^{n_-} fails for: %s" % (bad_sign if bad_sign else "nothing"))
    print("names whose prefix N does not match (non-)symmetrizability: %s"
          % (bad_name if bad_name else "none"))


if __name__ == "__main__":
    main()
