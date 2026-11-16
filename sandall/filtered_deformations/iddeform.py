#!/usr/bin/env python3
"""Are the three deforms of k^(1)(3; 1) pairwise non-isomorphic, and which are they?

er^(1)(3; 1) is identified in er3.py.  Here we only separate the deforms from one
another and from k^(1)(3; 1) itself, by isomorphism invariants that need no guess
about the answer:

    dim Der g,  dim H^1(g; g) = dim Der g - dim ad g,  dim [g, g],  dim Z(g),

and the dimensions of the Weisfeiler subalgebras already found by deform2.py.
"""
import numpy as np
from itertools import combinations

from k3 import P, rref, nullspace
from deform import cocycles, deform, g as k1


def table(alg):
    """the structure constants as a dense array c[i][j][k]."""
    n = alg.n
    C = np.zeros((n, n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k, c in alg.c[i][j].items():
                C[i, j, k] = c % P
    return C


def derivations(alg):
    """dim of Der g = {D : D[x,y] = [Dx,y] + [x,Dy]} over F_3."""
    n = alg.n
    C = table(alg)
    # unknowns D[a][b] (D e_b = sum_a D[a][b] e_a), n^2 of them
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                r = np.zeros(n * n, dtype=np.int64)
                # D([e_i,e_j])_k = sum_m c[i,j,m] D[k][m]
                for m in range(n):
                    if C[i, j, m]:
                        r[k * n + m] = (r[k * n + m] + C[i, j, m]) % P
                # - [D e_i, e_j]_k = - sum_a D[a][i] c[a,j,k]
                for a in range(n):
                    if C[a, j, k]:
                        r[a * n + i] = (r[a * n + i] - C[a, j, k]) % P
                # - [e_i, D e_j]_k = - sum_a D[a][j] c[i,a,k]
                for a in range(n):
                    if C[i, a, k]:
                        r[a * n + j] = (r[a * n + j] - C[i, a, k]) % P
                if r.any():
                    rows.append(r % P)
    M = np.array(rows, dtype=np.int64) if rows else np.zeros((1, n * n), np.int64)
    NS = nullspace([[int(a) for a in r] for r in M], n * n)
    return len(NS)


def centre(alg):
    n = alg.n
    C = table(alg)
    rows = []
    for j in range(n):
        for k in range(n):
            r = [int(C[i, j, k]) for i in range(n)]
            if any(r):
                rows.append(r)
    return len(nullspace(rows, n)) if rows else n


def derived(alg):
    n = alg.n
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            v = [0] * n
            for k, c in alg.c[i][j].items():
                v[k] = c % P
            if any(v):
                rows.append(v)
    return len(rref(rows, n)[0]) if rows else 0


def report(label, alg):
    z, dr, dd = centre(alg), derivations(alg), derived(alg)
    inv = (alg.n, z, dd, dr, dr - (alg.n - z))
    print("   %-26s dim g = %3d  dim Z = %d  dim [g,g] = %3d  "
          "dim Der = %3d  dim H^1(g;g) = %d" % ((label,) + inv))
    return inv


if __name__ == "__main__":
    print("=" * 78)
    print("isomorphism invariants of k^(1)(3; 1) and of its deforms")
    print("=" * 78)
    seen = {"k^(1)(3;1) itself": report("k^(1)(3;1) itself", k1)}
    for w in ((-6, 0), (-3, 3), (-3, -3)):
        idx = [i for i, (ww, k, c) in enumerate(cocycles) if ww == w and k == 2]
        if not idx:
            continue
        co = [0] * len(cocycles)
        co[idx[0]] = 1
        seen["deform along %s" % (w,)] = report("deform along %s" % (w,), deform(co))
    print()
    for a, b in combinations(seen, 2):
        verdict = "NOT isomorphic" if seen[a] != seen[b] else "invariants agree"
        print("   %-24s vs %-24s : %s" % (a, b, verdict))
