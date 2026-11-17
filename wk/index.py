#!/usr/bin/env python3
"""index.py -- Problem 4) of Sect. 10 and Statement 10.0.1 for o^{(1)}_I(2n+1): the index of the restricted
closure L and the generic stabilizer.

ind L = dim L - rank of the form chi([x,y]) for a generic chi in L^*.
The candidate for the maximum over ALL chi of the dimension of an irreducible
u(L; chi)-module is 2^{(dim L - ind L)/2}; the stabilizer of a generic chi is
tested for being a torus (abelian, and the 2-map x -> x^2 bijective on it).

Output: index.log
"""
import random
from wk import GF, oI, restricted_closure, nullspace, Space, mat_mul, flat

LOG = open("index.log", "w")


def out(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def rank(F, rows, ncols):
    S = Space(F, ncols)
    for r in rows:
        S.add(r)
    return S.rank()


def main():
    F = GF(6)
    random.seed(7)
    for n in (1, 2, 3):
        alg, deg = oI(F, n)
        L, odd = restricted_closure(alg)
        L.build_struct()
        d = L.n
        best = None
        for trial in range(3):
            chi = [random.randrange(F.q) for _ in range(d)]
            B = [[0] * d for _ in range(d)]
            for i in range(d):
                for j in range(d):
                    v = L.bracket(L._e(i) if hasattr(L, "_e") else
                                  [1 if t == i else 0 for t in range(d)],
                                  [1 if t == j else 0 for t in range(d)])
                    s = 0
                    for t in range(d):
                        if v[t]:
                            s = F.add(s, F.mul(v[t], chi[t]))
                    B[i][j] = s
            r = rank(F, B, d)
            if best is None or r > best[0]:
                best = (r, chi, B)
        r, chi, B = best
        ind = d - r
        stab = nullspace(F, B, d)
        # the stabilizer as matrices
        mats = []
        for v in stab:
            A = [[0] * L.m for _ in range(L.m)]
            for t in range(d):
                if v[t]:
                    for a in range(L.m):
                        for b in range(L.m):
                            if L.basis[t][a][b]:
                                A[a][b] = F.add(A[a][b], F.mul(v[t], L.basis[t][a][b]))
            mats.append(A)
        ab = all(mat_mul(F, X, Y) == mat_mul(F, Y, X) for X in mats for Y in mats)
        sq = [flat(mat_mul(F, X, X)) for X in mats]
        S = Space(F, L.m * L.m)
        for X in mats:
            S.add(flat(X))
        closed = all(v in S for v in sq)
        bij = rank(F, sq, L.m * L.m) == len(mats)
        out("n = %d: dim L = %d, ind L = %d, (dim L - ind L)/2 = %s; "
            "highest-weight exponent n(n+1) = %d; generic stabilizer: dim %d, "
            "abelian %s, closed under squaring %s, 2-map bijective %s"
            % (n, d, ind, (d - ind) / 2, n * (n + 1), len(stab), ab, closed, bij))


if __name__ == "__main__":
    main()
