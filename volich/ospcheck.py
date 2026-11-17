#!/usr/bin/env python3
"""
INDEPENDENT cross-check of the osp(4|2) part of nontriv.py.

Nothing here is imported from exc3.py/d21a.py: osp(4|2) is built from scratch
as the subalgebra of gl(4|2) preserving an even nondegenerate supersymmetric
form, by solving the defining linear condition over Q.

We then exhibit an odd x with [x,x]=0 (a root vector of an isotropic root) and
a, b in g_ev with [[a,x],[b,x]] != 0 -- i.e. x DOES ensure nontriviality in
the sense of eq. (31), as the last row of Table 2.12.1 says ('always').

Usage: python3 ospcheck.py
"""

from fractions import Fraction as F
from itertools import product

M, Nn = 4, 2          # sdim of the tautological module: 4|2
D = M + Nn


def par(i):
    return 0 if i < M else 1


# ---- the form: symmetric hyperbolic on V_0, symplectic on V_1
W = [[F(0)] * D for _ in range(D)]
W[0][2] = W[2][0] = F(1)
W[1][3] = W[3][1] = F(1)
W[4][5] = F(1)
W[5][4] = F(-1)


def mzero():
    return [[F(0)] * D for _ in range(D)]


def mmul(A, B):
    C = mzero()
    for i in range(D):
        for k in range(D):
            a = A[i][k]
            if a:
                for j in range(D):
                    if B[k][j]:
                        C[i][j] += a * B[k][j]
    return C


def madd(A, B, s=F(1)):
    return [[A[i][j] + s * B[i][j] for j in range(D)] for i in range(D)]


def mzerop(A):
    return all(not A[i][j] for i in range(D) for j in range(D))


def pmat(A):
    """parity of a homogeneous matrix, or None"""
    p = None
    for i in range(D):
        for j in range(D):
            if A[i][j]:
                q = (par(i) + par(j)) % 2
                if p is None:
                    p = q
                elif p != q:
                    return None
    return p


def sbr(A, B):
    """supercommutator AB - (-1)^{|A||B|} BA for homogeneous A,B"""
    pa, pb = pmat(A), pmat(B)
    sg = F(-1) ** ((0 if pa is None else pa) * (0 if pb is None else pb))
    return madd(mmul(A, B), mmul(B, A), -sg)


# ---- osp(4|2) = {X : w(Xu,v) + (-1)^{|X||u|} w(u,Xv) = 0}
def basis_of_parity(p):
    """solve the defining condition inside the parity-p part of gl(4|2)"""
    coords = [(i, j) for i in range(D) for j in range(D)
              if (par(i) + par(j)) % 2 == p]
    rows = []
    for u in range(D):
        for v in range(D):
            row = [F(0)] * len(coords)
            for k, (i, j) in enumerate(coords):
                # X = E_ij ; (Xu)_i = delta_ju  -> w(Xu,v) = delta_{j,u} W[i][v]
                if j == u:
                    row[k] += W[i][v]
                if j == v:
                    row[k] += F(-1) ** (p * par(u)) * W[u][i]
            if any(row):
                rows.append(row)
    # nullspace over Q
    m = [list(r) for r in rows]
    piv, r = [], 0
    ncols = len(coords)
    for c in range(ncols):
        s = None
        for i in range(r, len(m)):
            if m[i][c]:
                s = i
                break
        if s is None:
            continue
        m[r], m[s] = m[s], m[r]
        inv = F(1) / m[r][c]
        m[r] = [x * inv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c]:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        piv.append(c)
        r += 1
    free = [c for c in range(ncols) if c not in piv]
    out = []
    for fc in free:
        v = [F(0)] * ncols
        v[fc] = F(1)
        for ri, pc in enumerate(piv):
            v[pc] = -m[ri][fc]
        X = mzero()
        for k, (i, j) in enumerate(coords):
            X[i][j] = v[k]
        out.append(X)
    return out


def main():
    ev = basis_of_parity(0)
    od = basis_of_parity(1)
    print("osp(4|2) built from scratch in gl(4|2):")
    print("   sdim = (%d|%d)   [expected (9|8)]" % (len(ev), len(od)))
    assert (len(ev), len(od)) == (9, 8)

    # closed under the superbracket?
    allb = ev + od
    flat = [[X[i][j] for i in range(D) for j in range(D)] for X in allb]

    def inspan(X):
        tgt = [X[i][j] for i in range(D) for j in range(D)]
        rows = [list(r) + [F(0)] for r in flat]
        rows = [list(r) for r in flat] + [tgt]
        # rank test
        def rk(rs):
            m = [list(r) for r in rs]
            r = 0
            for c in range(len(m[0])):
                s = None
                for i in range(r, len(m)):
                    if m[i][c]:
                        s = i
                        break
                if s is None:
                    continue
                m[r], m[s] = m[s], m[r]
                inv = F(1) / m[r][c]
                m[r] = [x * inv for x in m[r]]
                for i in range(len(m)):
                    if i != r and m[i][c]:
                        f = m[i][c]
                        m[i] = [x - f * y for x, y in zip(m[i], m[r])]
                r += 1
                if r == len(m):
                    break
            return r
        return rk(flat) == rk(rows)

    bad = sum(1 for A in allb for B in allb if not inspan(sbr(A, B)))
    print("   closed under the superbracket: %s" % (bad == 0))
    assert bad == 0

    # ---- x = the odd element with A: f_1 |-> e_1  (e_1 isotropic in V_0)
    # find it inside the odd space: the unique (up to scalar) odd basis
    # element whose only nonzero V_1 -> V_0 entry is (0,4).
    x = None
    for X in od:
        if X[0][4] and not any(X[i][4] for i in range(1, M)) \
                and not any(X[i][5] for i in range(M)):
            x = X
            break
    if x is None:
        # take a combination
        for X in od:
            if X[0][4]:
                x = X
                break
    assert x is not None
    print("\n   x =")
    for r in x:
        print("        [" + "  ".join("%4s" % str(t) for t in r) + "]")

    xx = sbr(x, x)
    print("   [x,x] = 0 : %s   (x is homologic, an isotropic odd root vector)"
          % mzerop(xx))
    assert mzerop(xx)

    found = []
    for ia, A in enumerate(ev):
        for ib, B in enumerate(ev):
            u, v = sbr(A, x), sbr(B, x)
            c = sbr(u, v)
            if not mzerop(c):
                found.append((ia, ib, A, B, c))
    print("   pairs (a,b) in a basis of g_ev with [[a,x],[b,x]] != 0 : %d of %d"
          % (len(found), len(ev) ** 2))
    if found:
        ia, ib, A, B, c = found[0]
        print("\n   witness: a = e_%d, b = e_%d" % (ia, ib))
        print("   a =")
        for r in A:
            print("        [" + "  ".join("%4s" % str(t) for t in r) + "]")
        print("   b =")
        for r in B:
            print("        [" + "  ".join("%4s" % str(t) for t in r) + "]")
        print("   [[a,x],[b,x]] =")
        for r in c:
            print("        [" + "  ".join("%4s" % str(t) for t in r) + "]")
    print("\n   => x ENSURES NONTRIVIALITY in osp(4|2).")
    print("      As the last row of Table 2.12.1 says ('always').")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
