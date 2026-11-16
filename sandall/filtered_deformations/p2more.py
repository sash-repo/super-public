"""Further p=2 cases, and the two isomorphism claims of Caranti-Mattarei.

(a) dim H^2(H(2:n;omega_0);F) at p=2 for more n, to confirm the corrected
    count n1+n2-1 (Remark 4.5 of arXiv:math/0510431 says n1+n2).
(b) H(2:(1,n);omega_2) ~ the simple Zassenhaus algebra W(1:n+1)^(1) at p=2,
    by the map of Section 3: x y^(j) |-> E_{j-1},  y^(j) |-> E_{j+2^n-2}.
(c) H(2:(1,1);omega_0) ~ psl(3) at p=3 (Remark 4.3), by dimension and by
    Der being 14-dimensional (type G_2).
"""
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank                                     # noqa: E402
from nongraded import Ham                                     # noqa: E402


def h2_only(p, n1, n2, form):
    L = Ham(p, n1, n2, form)
    L.table()
    h2, z2, b2 = L.h2_dim()
    print("p=%d H(2:(%d,%d);om_%d) dim=%2d : dim H^2 = %d   [n1+n2 = %d, n1+n2-1 = %d]"
          % (p, n1, n2, form, L.dim, h2, n1 + n2, n1 + n2 - 1))
    return h2


# ---------------------------------------------------------------- Zassenhaus

def zassenhaus_simple(p, n):
    """W(1:n)^(1) at p=2: basis E_i = x^(i+1) d/dx, i = -1..r, r = p^n-2,
    minus E_r.  [E_i,E_j] = (C(i+j+1,j) - C(i+j+1,i)) E_{i+j}."""
    from nongraded import binom
    idx = [i for i in range(-1, p ** n - 1) if i != p ** n - 2]
    pos = {i: t for t, i in enumerate(idx)}
    d = len(idx)
    T = [[dict() for _ in range(d)] for _ in range(d)]
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            c = (binom(i + j + 1, j) - binom(i + j + 1, i)) % p
            if c and (i + j) in pos:
                T[a][b][pos[i + j]] = c
    return idx, pos, T


def check_zassenhaus_iso(n2):
    """H(2:(1,n2);om_2) at p=2 versus the simple Zassenhaus W(1:n2+1)^(1)."""
    p = 2
    L = Ham(p, 1, n2, 2)
    L.table()
    idx, pos, T = zassenhaus_simple(p, n2 + 1)
    if len(idx) != L.dim:
        print("  dimensions differ: %d vs %d" % (L.dim, len(idx)))
        return
    # x y^(j) |-> E_{j-1};  y^(j) |-> E_{j + 2^n2 - 2}
    phi = {}
    for a, (i, j) in enumerate(L.basis):
        e = (j - 1) if i == 1 else (j + 2 ** n2 - 2)
        if i not in (0, 1) or e not in pos:
            print("  the map is undefined on x^(%d)y^(%d)" % (i, j))
            return
        phi[a] = pos[e]
    if len(set(phi.values())) != L.dim:
        print("  the map is not bijective")
        return
    bad = 0
    for a in range(L.dim):
        for b in range(L.dim):
            lhs = {phi[k]: v for k, v in L._table[a][b].items()}
            rhs = T[phi[a]][phi[b]]
            if lhs != rhs:
                bad += 1
    print("p=2 H(2:(1,%d);om_2) dim=%d vs simple Zassenhaus W(1:%d)^(1) dim=%d : "
          "%s" % (n2, L.dim, n2 + 1, len(idx),
                  "isomorphic (the paper's map)" if bad == 0
                  else "the paper's map fails on %d pairs" % bad))


# ---------------------------------------------------------------- psl(3), p=3

def check_psl3():
    p = 3
    L = Ham(p, 1, 1, 0)
    L.table()
    # sl(3) mod its centre at p=3
    import itertools
    basisM = []
    for a, b in itertools.product(range(3), repeat=2):
        if a != b:
            basisM.append(("E", a, b))
    basisM.append(("H", 0, 0))       # E11-E22
    basisM.append(("H", 1, 1))       # E22-E33
    # the centre of sl(3) at p=3 is the identity = H1 - H2 direction? build matrices
    def mat(spec):
        M = [[0] * 3 for _ in range(3)]
        if spec[0] == "E":
            M[spec[1]][spec[2]] = 1
        else:
            k = spec[1]
            M[k][k], M[k + 1][k + 1] = 1, -1 % p
        return M
    mats = [mat(s) for s in basisM]
    flat = [sum(M, []) for M in mats]
    ident = [1 if r == c else 0 for r in range(3) for c in range(3)]
    print("p=3: dim sl(3) = %d, is the identity in sl(3)? %s ; dim psl(3) = %d, "
          "dim H(2:(1,1);om_0) = %d"
          % (rank(flat, p=p), rank(flat + [ident], p=p) == rank(flat, p=p),
             rank(flat, p=p) - 1, L.dim))


if __name__ == "__main__":
    print("-- (a) H^2 of the graded algebra at p=2")
    for (n1, n2) in [(2, 2), (2, 3), (3, 3), (1, 2), (1, 3)]:
        h2_only(2, n1, n2, 0)
    print()
    print("-- (b) the p=2 Zassenhaus identification")
    for n2 in (1, 2, 3):
        check_zassenhaus_iso(n2)
    print()
    print("-- (c) p=3, n=(1,1)")
    check_psl3()
