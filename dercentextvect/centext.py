#!/usr/bin/env python3
"""Central extensions: H^2(L; K) for the modular vectorial Lie superalgebras.

The other half of the subject: `dervect.py` computes Der L, this one computes
the (even and odd) central extensions, i.e. the Chevalley--Eilenberg
cohomology H^2(L; K) with the trivial coefficients.

Conventions.  A 2-cochain is a bilinear form c on L with

    c(y, x) = -(-1)^{p(x)p(y)} c(x, y),

homogeneous of parity eps: c(x, y) = 0 unless p(x) + p(y) = eps.  Thus an even
cochain lives on Lambda^2 L_ev + S^2 L_od and an odd one on L_ev (x) L_od.
The cocycle condition is the one that makes L + K z with
[x, y]' := [x, y] + c(x, y) z a Lie superalgebra:

    c([x, y], z) - c(x, [y, z]) + (-1)^{p(x)p(y)} c(y, [x, z]) = 0;

the coboundary of f in L^* is (df)(x, y) = f([x, y]).

Everything splits by the Z-grading: the differential preserves the weight, so
H^2 = (+)_w H^2_w and each block is small.  `h2_ungraded` recomputes the same
number with no grading used, as a cross-check.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exactlin import rank, nullspace                              # noqa: E402
from dervect import Super, Echelon, homogeneous_basis, Struct     # noqa: E402


class Cocycles:
    """H^2(L; K) for a graded Lie superalgebra L given by spanning elements."""

    def __init__(self, S, L):
        self.S, self.p = S, S.p
        hom = homogeneous_basis(S, L)
        self.st = Struct(S, hom)
        self.N = self.st.N
        self.deg = self.st.deg
        self.par = self.st.par

    # ------------------------------------------------------------- cochains
    def pairs(self, w=None, eps=None):
        """the coordinates of a 2-cochain of weight w and parity eps."""
        out = []
        for i in range(self.N):
            for j in range(i, self.N):
                if i == j and self.par[i] == 0:
                    continue          # c(x, x) = -c(x, x) and p is odd
                if w is not None and self.deg[i] + self.deg[j] != w:
                    continue
                if eps is not None and (self.par[i] + self.par[j]) % 2 != eps:
                    continue
                out.append((i, j))
        return out

    def coord_of(self, idx, i, j):
        """(index in the unknown vector, sign) of the value c(b_i, b_j)."""
        if (i, j) in idx:
            return idx[(i, j)], 1
        if (j, i) in idx:
            sg = -((-1) ** (self.par[i] * self.par[j]))
            return idx[(j, i)], sg % self.p
        return None, 0

    # ------------------------------------------------------------ the block
    def block(self, w, eps, all_triples=False):
        """(dim Z^2, dim B^2) in the block of weight w and parity eps."""
        P = self.pairs(w, eps)
        if not P:
            return 0, 0
        idx = {q: t for t, q in enumerate(P)}
        n = len(P)
        p = self.p
        cc = self.st.cc

        def add_term(row, coef, l, k):
            """row += coef * c(b_l, b_k)."""
            t, sg = self.coord_of(idx, l, k)
            if t is not None and sg:
                row[t] = (row[t] + coef * sg) % p

        eqs = []
        rng = range(self.N)
        for i in rng:
            for j in (rng if all_triples else range(i, self.N)):
                dij, pij = self.deg[i] + self.deg[j], self.par[i] + self.par[j]
                cij = cc(i, j)
                for k in (rng if all_triples else range(j, self.N)):
                    if dij + self.deg[k] != w:
                        continue
                    if (pij + self.par[k]) % 2 != eps:
                        continue
                    row = [0] * n
                    # c([b_i, b_j], b_k)
                    for l in rng:
                        if cij[l]:
                            add_term(row, cij[l], l, k)
                    # - c(b_i, [b_j, b_k])
                    cjk = cc(j, k)
                    for l in rng:
                        if cjk[l]:
                            add_term(row, -cjk[l], i, l)
                    # + (-1)^{p_i p_j} c(b_j, [b_i, b_k])
                    sg = (-1) ** (self.par[i] * self.par[j])
                    cik = cc(i, k)
                    for l in rng:
                        if cik[l]:
                            add_term(row, sg * cik[l], j, l)
                    if any(row):
                        eqs.append(row)
        dimZ = n - rank(eqs, p=p) if eqs else n

        # coboundaries: f dual to b_l with deg = w, par = eps
        cob = []
        for l in rng:
            if self.deg[l] != w or self.par[l] != eps:
                continue
            row = [0] * n
            for t, (i, j) in enumerate(P):
                row[t] = cc(i, j)[l] % p
            if any(row):
                cob.append(row)
        dimB = rank(cob, p=p) if cob else 0
        return dimZ, dimB

    # ---------------------------------------------------------------- total
    def h2(self, all_triples=False):
        """{(w, eps): dim H^2_{w, eps}} over the nonzero blocks, and the totals."""
        ws = sorted({self.deg[i] + self.deg[j]
                     for i in range(self.N) for j in range(self.N)})
        blocks, tot = {}, [0, 0]
        for w in ws:
            for eps in (0, 1):
                dZ, dB = self.block(w, eps, all_triples)
                if dZ - dB:
                    blocks[(w, eps)] = dZ - dB
                    tot[eps] += dZ - dB
        return blocks, tuple(tot)


def h2_ungraded(S, L):
    """the same H^2, with the grading not used at all (a cross-check)."""
    C = Cocycles(S, L)
    P = C.pairs()
    idx = {q: t for t, q in enumerate(P)}
    n, p, cc = len(P), C.p, C.st.cc

    def term(row, coef, l, k):
        t, sg = C.coord_of(idx, l, k)
        if t is not None and sg:
            row[t] = (row[t] + coef * sg) % p

    eqs = []
    for i in range(C.N):
        for j in range(i, C.N):
            cij = cc(i, j)
            for k in range(j, C.N):
                row = [0] * n
                for l in range(C.N):
                    if cij[l]:
                        term(row, cij[l], l, k)
                cjk = cc(j, k)
                for l in range(C.N):
                    if cjk[l]:
                        term(row, -cjk[l], i, l)
                sg = (-1) ** (C.par[i] * C.par[j])
                cik = cc(i, k)
                for l in range(C.N):
                    if cik[l]:
                        term(row, sg * cik[l], j, l)
                if any(row):
                    eqs.append(row)
    dimZ = n - rank(eqs, p=p) if eqs else n
    cob = []
    for l in range(C.N):
        row = [cc(i, j)[l] % p for (i, j) in P]
        if any(row):
            cob.append(row)
    return dimZ - (rank(cob, p=p) if cob else 0)


def cocycle_reps(S, L, w, eps):
    """a basis of H^2_{w, eps}, each cocycle as {(i, j): value} on the basis."""
    C = Cocycles(S, L)
    P = C.pairs(w, eps)
    if not P:
        return C, []
    idx = {q: t for t, q in enumerate(P)}
    n, p, cc = len(P), C.p, C.st.cc

    def term(row, coef, l, k):
        t, sg = C.coord_of(idx, l, k)
        if t is not None and sg:
            row[t] = (row[t] + coef * sg) % p

    eqs = []
    for i in range(C.N):
        for j in range(i, C.N):
            cij = cc(i, j)
            for k in range(j, C.N):
                if C.deg[i] + C.deg[j] + C.deg[k] != w:
                    continue
                if (C.par[i] + C.par[j] + C.par[k]) % 2 != eps:
                    continue
                row = [0] * n
                for l in range(C.N):
                    if cij[l]:
                        term(row, cij[l], l, k)
                cjk = cc(j, k)
                for l in range(C.N):
                    if cjk[l]:
                        term(row, -cjk[l], i, l)
                sg = (-1) ** (C.par[i] * C.par[j])
                cik = cc(i, k)
                for l in range(C.N):
                    if cik[l]:
                        term(row, sg * cik[l], j, l)
                if any(row):
                    eqs.append(row)
    Z = nullspace(eqs, p=p) if eqs else [[1 if t == s else 0 for t in range(n)]
                                         for s in range(n)]
    cob = []
    for l in range(C.N):
        if C.deg[l] != w or C.par[l] != eps:
            continue
        row = [cc(i, j)[l] % p for (i, j) in P]
        if any(row):
            cob.append(row)
    E = Echelon([[int(x) % p for x in r] for r in cob], p) if cob else None
    reps = []
    seen = Echelon([[int(x) % p for x in r] for r in cob], p) if cob else Echelon([], p)
    for v in Z:
        vv = [int(x) % p for x in v]
        if seen.contains(vv):
            continue
        seen = Echelon([r for r in seen.rows] + [vv], p)
        reps.append({P[t]: vv[t] for t in range(n) if vv[t]})
    return C, reps


if __name__ == "__main__":
    # a smoke test: the Witt algebra vect(1; 1) and its p=3, 5 relatives
    for p in (3, 5, 7):
        S = Super(p, 1, 0, [1])
        C = Cocycles(S, S.vect())
        b, tot = C.h2()
        print(f"p={p} vect(1;1): dim L={C.N} H^2 blocks={b} totals={tot} "
              f"ungraded={h2_ungraded(S, S.vect())}")
