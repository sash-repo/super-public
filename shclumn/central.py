"""Calibration of the window method against a KNOWN nonzero answer:
H^2(g; C), the central extensions of the distinguished stringy Lie
superalgebras, classified in [KvdL].

"Distinguished" MEANS that this space is nonzero, so every algebra in the list
must return at least 1 here, and the polynomial counterparts must return 0
(vect(1) = W_{>=-1} has no Virasoro cocycle).  If the window method reproduces
that, the same machinery applied to the adjoint module is to be believed.

Trivial coefficients: c: Lambda^2 g -> C, weight 0 means c(g_i,g_j) = 0 unless
i + j = 0; the cocycle identity is the one of defco.py with [c(x,y),z] dropped
(the module is trivial), and (delta phi)(x,y) = -phi([x,y]) for phi a linear
form on g_0.
"""

from fractions import Fraction as Q
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from exactlin import rank, nullspace
from defco import Graded, algebras
import yb


def h2_trivial(g, N):
    """dim H^2_{(0)}(g; C) computed in a window of N steps."""
    degs = g.degrees(N)
    win = set(degs)
    vec = []
    for d in degs:
        for v, p in g.basis(d):
            vec.append((v, p, d))
    nv = len(vec)

    # ---- unknowns: the pairs i <= j of opposite degree and equal parity
    unk = {}
    for i in range(nv):
        vi, pi, di = vec[i]
        for j in range(i, nv):
            vj, pj, dj = vec[j]
            if di + dj != 0 or pi != pj:
                continue
            if i == j and pi == 0:
                continue                       # c(x,x) = 0 for x even
            unk[(i, j)] = len(unk)

    def cv(i, j):
        """c(e_i,e_j) as a linear form {unknown -> coefficient}."""
        s = Q(1)
        if j < i:
            pi, pj = vec[i][1], vec[j][1]
            s = -Q((-1) ** (pi * pj))
            i, j = j, i
        return {unk[(i, j)]: s} if (i, j) in unk else {}

    def cv_elt(elt, d, j):
        """c(v, e_j) for a homogeneous element v of degree d."""
        out = {}
        if d not in win:
            return None
        co = g.coords(d, elt)
        idx = [t for t in range(nv) if vec[t][2] == d]
        for a, t in enumerate(idx):
            if not co[a]:
                continue
            for u, c in cv(t, j).items():
                out[u] = out.get(u, Q(0)) + co[a] * c
        return out

    # ---- equations
    rows = []
    for i in range(nv):
        for j in range(i, nv):
            for k in range(j, nv):
                if vec[i][2] + vec[j][2] + vec[k][2] != 0:
                    continue
                if (vec[i][2] + vec[j][2] not in win or
                        vec[j][2] + vec[k][2] not in win or
                        vec[i][2] + vec[k][2] not in win):
                    continue
                acc, bad = {}, False
                for (x, y, z) in ((i, j, k), (j, k, i), (k, i, j)):
                    px, pz = vec[x][1], vec[z][1]
                    s = Q((-1) ** (px * pz))
                    b = g.brk(vec[x][0], vec[y][0])
                    if not b:
                        continue
                    got = cv_elt(b, vec[x][2] + vec[y][2], z)
                    if got is None:
                        bad = True
                        break
                    for u, c in got.items():
                        acc[u] = acc.get(u, Q(0)) + s * c
                if bad:
                    continue
                acc = {u: c for u, c in acc.items() if c}
                if acc:
                    rows.append(acc)

    n = len(unk)
    if n == 0:
        return 0, 0, 0
    mat = [[r.get(c, Q(0)) for c in range(n)] for r in rows]
    Z = nullspace(mat) if mat else [[Q(1) if a == b else Q(0)
                                       for b in range(n)] for a in range(n)]
    # ---- coboundaries: phi a linear form on g_0
    B = []
    zero = [t for t in range(nv) if vec[t][2] == 0]
    for t in zero:
        row = [Q(0)] * n
        ok = True
        for (i, j), u in unk.items():
            b = g.brk(vec[i][0], vec[j][0])
            if not b:
                continue
            co = g.coords(0, b)
            idx = [a for a in range(nv) if vec[a][2] == 0]
            row[u] = -co[idx.index(t)]
        if ok:
            B.append(row)
    rZ, rB = (rank(Z) if Z else 0), (rank(B) if B else 0)
    return rZ - rB, rZ, rB


def main():
    print("H^2(g; C) -- the central extensions, window method\n")
    print("  %-16s %-6s %-6s %-6s %s" % ("algebra", "N", "Z", "B", "h"))
    for laur in (True, False):
        print("\n  %s coefficients" % ("Laurent" if laur else "polynomial"))
        for g in algebras(laurent=laur):
            row = []
            for N in (3, 4, 5):
                h, rZ, rB = h2_trivial(g, N)
                row.append((N, rZ, rB, h))
            N, rZ, rB, h = row[-1]
            print("  %-16s %-6d %-6d %-6d %d      (N=3,4,5: %s)"
                  % (g.name, N, rZ, rB, h, ",".join(str(r[3]) for r in row)))


if __name__ == "__main__":
    main()
