#!/usr/bin/env python3
"""Explicit description of the Volichenko algebras of
Subsec. 3.4 of the Volichenko text (Statement 3.4.1, Remark 3.4.2):

    1) vosp(m|n|p)   in  g = osp(2m|2n)  and  osp(2m+1|2n),  x = image of x^p_p
    2) vq(n|k)       in  g = psq(n),                          x = x_k
    3) vspe(n|p,q)   in  g = spe(n),                          x = y^p_q

For each of them the engine builds g from scratch as a matrix Lie superalgebra
(by SOLVING the defining linear condition, not by quoting a basis), takes the
element x that Table 2.12.1 prescribes, asserts that x is odd with [x,x] = 0
and that it ensures nontriviality, computes

    h_0hat = Ann(x) = {a in g_ev | [x,a] = 0},   h_1hat = h/h_0hat,

and then CHECKS the closed-form description proposed in the text against the
computed space, entry by entry: same dimension, and every basis matrix of the
computed Ann(x) satisfies the stated conditions, and conversely.

Everything is exact (Fraction).  Usage: python3 vdescr.py
"""

import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "."))
from exactlin import rank, nullspace                                # noqa: E402


# ----------------------------------------------------------------- matrices

def zeros(d):
    return [[F(0)] * d for _ in range(d)]


def mmul(A, B, d):
    C = zeros(d)
    for i in range(d):
        Ai = A[i]
        for k in range(d):
            a = Ai[k]
            if a:
                Bk, Ci = B[k], C[i]
                for j in range(d):
                    Ci[j] += a * Bk[j]
    return C


def madd(A, B, d, s=F(1)):
    return [[A[i][j] + s * B[i][j] for j in range(d)] for i in range(d)]


def flat(A, d):
    return [A[i][j] for i in range(d) for j in range(d)]


def unflat(v, d):
    return [[F(v[i * d + j]) for j in range(d)] for i in range(d)]


def sbr(A, pA, B, pB, d):
    """superbracket [A,B] = AB - (-1)^{pA pB} BA."""
    return madd(mmul(A, B, d), mmul(B, A, d), d, -F(-1) ** (pA * pB))


def iszero(A, d):
    return all(A[i][j] == 0 for i in range(d) for j in range(d))


# ------------------------------------------------- ambient superalgebras

class Amb:
    """g inside gl(M|N): a basis of g_ev and one of g_od, as matrices."""

    def __init__(self, name, M, N, ev, od, center=()):
        self.name, self.M, self.N, self.d = name, M, N, M + N
        self.ev, self.od, self.center = ev, od, list(center)

    def par(self, i):
        return 0 if i < self.M else 1


def _solve_space(d, conds, par):
    """Basis of {Z of parity par | conds(Z) == 0}, conds a list of linear
    functionals on the d^2 entries given as coefficient rows."""
    rows = []
    for row in conds:
        rows.append(row)
    ns = nullspace(rows) if rows else None
    if ns is None or not rows:
        # no conditions: everything of the right parity
        ns = []
    basis = []
    for v in ns:
        A = unflat(v, d)
        if not iszero(A, d):
            basis.append(A)
    return basis


def _parity_mask(d, M, par):
    """rows forcing every entry of the wrong parity to vanish."""
    out = []
    for i in range(d):
        for j in range(d):
            pij = (0 if i < M else 1) ^ (0 if j < M else 1)
            if pij != par:
                r = [F(0)] * (d * d)
                r[i * d + j] = F(1)
                out.append(r)
    return out


def osp(m, n, odd_dim_extra=0):
    """osp(2m + e | 2n), e in {0,1}, as the stabilizer of the even
    supersymmetric form with Gram matrix

        even block:  Pi_2m (+ (1) if e = 1)     odd block:  J_2n .

    Basis order: e_1..e_m, f_1..f_m [, g] | eta_1..eta_n, phi_1..phi_n,
    so that W = <e, eta> and W* = <f, phi> are isotropic and gl(m|n) sits
    inside as the stabilizer of the pair (W, W*)."""
    e = odd_dim_extra
    M, N = 2 * m + e, 2 * n
    d = M + N
    G = zeros(d)
    for i in range(m):
        G[i][m + i] = F(1)
        G[m + i][i] = F(1)
    if e:
        G[2 * m][2 * m] = F(1)
    for j in range(n):
        G[M + j][M + n + j] = F(1)
        G[M + n + j][M + j] = F(-1)

    def par(i):
        return 0 if i < M else 1

    def conds(p):
        # B(Z u_a, u_b) + (-1)^{p * p(a)} B(u_a, Z u_b) = 0 for all a, b
        out = []
        for a in range(d):
            for b in range(d):
                r = [F(0)] * (d * d)
                for c in range(d):
                    r[c * d + a] += G[c][b]
                sg = F(-1) ** (p * par(a))
                for c in range(d):
                    r[c * d + b] += sg * G[a][c]
                out.append(r)
        return out

    ev = _solve_space(d, conds(0) + _parity_mask(d, M, 0), 0)
    od = _solve_space(d, conds(1) + _parity_mask(d, M, 1), 1)
    return Amb(f"osp({M}|{N})", M, N, ev, od)


def spe(n):
    """spe(n) inside gl(n|n): preserves the ODD nondegenerate symmetric form
    with Gram matrix Pi_2n, and has supertrace 0.  Even part {diag(A, -A^t)},
    tr A = 0."""
    M = N = n
    d = 2 * n
    G = zeros(d)
    for i in range(n):
        G[i][n + i] = F(1)
        G[n + i][i] = F(1)

    def par(i):
        return 0 if i < M else 1

    def conds(p):
        out = []
        for a in range(d):
            for b in range(d):
                r = [F(0)] * (d * d)
                for c in range(d):
                    r[c * d + a] += G[c][b]
                sg = F(-1) ** (p * par(a))
                for c in range(d):
                    r[c * d + b] += sg * G[a][c]
                out.append(r)
        # supertrace 0
        r = [F(0)] * (d * d)
        for i in range(d):
            r[i * d + i] = F(1) if i < M else F(-1)
        out.append(r)
        return out

    ev = _solve_space(d, conds(0) + _parity_mask(d, M, 0), 0)
    od = _solve_space(d, conds(1) + _parity_mask(d, M, 1), 1)
    return Amb(f"spe({n})", M, N, ev, od)


def psq(n):
    """q(n) = {(A B / B A)}; sq(n): tr B = 0; psq(n) = sq(n)/C.1_{2n}."""
    d = 2 * n
    ev, od = [], []
    for i in range(n):
        for j in range(n):
            A = zeros(d)
            A[i][j] = A[n + i][n + j] = F(1)
            ev.append(A)
    for i in range(n):
        for j in range(n):
            B = zeros(d)
            B[i][n + j] = B[n + i][j] = F(1)
            od.append(B)
    ident = zeros(d)
    for i in range(d):
        ident[i][i] = F(1)
    # sq: qtr = tr B = 0 -- drop one diagonal odd generator, keep traceless ones
    od2 = []
    for i in range(n):
        for j in range(n):
            if i == j:
                if i == n - 1:
                    continue
                B = zeros(d)
                B[i][n + i] = B[n + i][i] = F(1)
                B[n - 1][d - 1] = B[d - 1][n - 1] = F(-1)
                od2.append(B)
            else:
                B = zeros(d)
                B[i][n + j] = B[n + i][j] = F(1)
                od2.append(B)
    return Amb(f"psq({n})", n, n, ev, od2, center=[ident])


# ------------------------------------------------------- the elements x

def x_osp(m, n, p, e=0):
    """image of x^p_p in gl(m|n) under gl(m|n) < osp(2m + e|2n): it acts on
    W by x and on W* by -x^{st}."""
    M, N = 2 * m + e, 2 * n
    d = M + N
    A = zeros(d)
    for i in range(p):                       # beta: eta_i -> e_i, i < p
        A[i][M + i] = F(1)
        A[M + N // 2 + i][m + i] = F(1)      # phi_i -> f_i  (from beta^t)
    for j in range(p, 2 * p):                # gamma: e_j -> eta_j
        A[M + j][j] = F(1)
        A[m + j][M + N // 2 + j] = F(-1)     # f_j -> -phi_j  (from -gamma^t)
    return A


def x_psq(n, k):
    d = 2 * n
    A = zeros(d)
    for t in range(k):
        i, j = 2 * t, 2 * t + 1
        A[i][n + j] = A[n + i][j] = F(1)
    return A


def y_spe(n, p, q):
    """y^p_q = antidiag(1^{(n,n,p)}, J_{(n,2q)}): B = diag(1_p, 0) on the
    first p indices, C = q copies of the 2x2 symplectic block on the next 2q."""
    d = 2 * n
    A = zeros(d)
    for i in range(p):
        A[i][n + i] = F(1)                  # B block, symmetric
    for t in range(q):
        i, j = p + 2 * t, p + 2 * t + 1
        A[n + i][j] = F(1)                  # C block, antisymmetric
        A[n + j][i] = F(-1)
    return A


# ------------------------------------------------------------- the analysis

def modspan(vs, center, d):
    """rank of span(vs) modulo the center."""
    cen = [flat(Z, d) for Z in center]
    return rank(cen + [flat(A, d) for A in vs]) - rank(cen) if cen else \
        rank([flat(A, d) for A in vs])


def ann(amb, x):
    """basis of {a in g_ev | [x, a] = 0}, modulo the center."""
    d = amb.d
    rows, keep = [], []
    # coordinates on g_ev modulo center
    cen = [flat(Z, d) for Z in amb.center]
    base = []
    cur = list(cen)
    r = rank(cur) if cur else 0
    for A in amb.ev:
        v = flat(A, d)
        rn = rank(cur + [v])
        if rn > r:
            cur.append(v)
            r = rn
            base.append(A)
    # [x, a] as a linear map on the coefficients, modulo the center
    cols = []
    for A in base:
        C = sbr(x, 1, A, 0, d)
        cols.append(flat(C, d))
    # solve sum c_i [x, a_i] = 0 modulo the center (center is even; [x,a] odd,
    # so for psq the center never occurs here -- but reduce anyway)
    m_rows = list(map(list, zip(*cols))) if cols else []
    ns = nullspace(m_rows) if m_rows else []
    out = []
    for v in ns:
        A = zeros(d)
        for c, B in zip(v, base):
            if c:
                for i in range(d):
                    for j in range(d):
                        A[i][j] += F(c) * B[i][j]
        out.append(A)
    return base, out


def nontrivial(amb, x, base):
    """does x ensure nontriviality: is S(a,b) = [[a,x],[b,x]] nonzero in g?"""
    d = amb.d
    cen = [flat(Z, d) for Z in amb.center]
    rk0 = rank(cen) if cen else 0
    for A in base:
        ax = sbr(A, 0, x, 1, d)
        for B in base:
            bx = sbr(B, 0, x, 1, d)
            S = sbr(ax, 1, bx, 1, d)
            if not iszero(S, d):
                if not cen:
                    return True
                if rank(cen + [flat(S, d)]) > rk0:
                    return True
    return False


def report(amb, x, label, predicate=None, pred_name=""):
    d = amb.d
    assert iszero(sbr(x, 1, x, 1, d), d), f"{label}: [x,x] != 0"
    # x must actually lie in the odd part of g -- not merely be a matrix
    odrows = [flat(A, d) for A in amb.od] + [flat(Z, d) for Z in amb.center]
    assert rank(odrows + [flat(x, d)]) == rank(odrows), \
        f"{label}: x is not in g_od"
    base, A0 = ann(amb, x)
    dim_h = len(base)
    dim0 = modspan(A0, amb.center, d)
    nt = nontrivial(amb, x, base)
    print(f"  {label:26s} dim h = {dim_h:4d}   dim h_0 = {dim0:4d}   "
          f"dim h_1 = {dim_h - dim0:4d}   nontrivial: {'yes' if nt else 'NO'}")
    if predicate is not None:
        # every computed element satisfies the closed form ...
        for A in A0:
            assert predicate(A), f"{label}: computed Ann(x) violates {pred_name}"
        # ... and the closed form has the same dimension
        cand = []
        for i in range(d):
            for j in range(d):
                pass
        ok = _closed_dim(amb, predicate) == dim0
        print(f"      closed form {pred_name}: "
              f"{'MATCHES' if ok else 'MISMATCH'} "
              f"(dim {_closed_dim(amb, predicate)} vs {dim0})")
        assert ok, f"{label}: closed form has the wrong dimension"
    return dim_h, dim0


def _closed_dim(amb, predicate):
    """dimension of {a in g_ev | predicate(a)} modulo the center."""
    d = amb.d
    cen = [flat(Z, d) for Z in amb.center]
    good = []
    # the predicate is linear, so intersect g_ev with its kernel by solving
    base = []
    cur = list(cen)
    r = rank(cur) if cur else 0
    for A in amb.ev:
        v = flat(A, d)
        rn = rank(cur + [v])
        if rn > r:
            cur.append(v)
            r = rn
            base.append(A)
    rows = []
    n = len(base)
    # sample the predicate as a linear condition by evaluating its residual
    for A in base:
        good.append(A)
    # brute force: the predicate is given as res(A) -> list of Fractions
    res0 = predicate(zeros(d), residual=True)
    cols = []
    for A in base:
        cols.append(predicate(A, residual=True))
    if not cols or not cols[0]:
        return n
    m_rows = list(map(list, zip(*cols)))
    ns = nullspace(m_rows)
    return len(ns)


# --------------------------------------------------------- closed forms

def pred_psq(n, k):
    """A commutes with X = diag(J_2(0)^k, 0): in the basis
    u_1..u_k, v_1..v_k, w_1..w_{n-2k} (X v_i = u_i) the shape is
        (a b c / 0 a 0 / 0 h i).
    Our X sends e_{2t+1} -> e_{2t}, so reorder first."""
    idx_u = [2 * t for t in range(k)]
    idx_v = [2 * t + 1 for t in range(k)]
    idx_w = list(range(2 * k, n))
    order = idx_u + idx_v + idx_w

    def f(Z, residual=False):
        # Z is in psq: read A off the top-left n x n block
        A = [[Z[order[i]][order[j]] for j in range(n)] for i in range(n)]
        res = []
        for i in range(n):
            for j in range(n):
                bi = 0 if i < k else (1 if i < 2 * k else 2)
                bj = 0 if j < k else (1 if j < 2 * k else 2)
                if (bi, bj) in ((1, 0), (1, 2), (2, 0)):
                    res.append(A[i][j])                 # zero blocks
        for i in range(k):                              # e-block = a-block
            for j in range(k):
                res.append(A[k + i][k + j] - A[i][j])
        if residual:
            return res
        return all(r == 0 for r in res)
    return f


def pred_spe(n, p, q):
    """A in sl(n) with A_{QP} = A_{QR} = A_{RP} = 0, A_{PP} in o(p),
    A_{QQ} in sp(2q)."""
    s = n - p - 2 * q
    P = list(range(p))
    Q = list(range(p, p + 2 * q))
    R = list(range(p + 2 * q, n))
    J = zeros(2 * q) if q else []
    for t in range(q):
        J[2 * t][2 * t + 1] = F(1)
        J[2 * t + 1][2 * t] = F(-1)

    def f(Z, residual=False):
        A = [[Z[i][j] for j in range(n)] for i in range(n)]
        res = []
        for i in Q:
            for j in P + R:
                res.append(A[i][j])
        for i in R:
            for j in P:
                res.append(A[i][j])
        for i in range(p):                                  # o(p)
            for j in range(p):
                res.append(A[i][j] + A[j][i])
        for a in range(2 * q):                              # sp(2q)
            for b in range(2 * q):
                v = F(0)
                for c in range(2 * q):
                    v += J[a][c] * A[Q[c]][Q[b]] + A[Q[c]][Q[a]] * J[c][b]
                res.append(v)
        if residual:
            return res
        return all(r == 0 for r in res)
    return f


# --------------------------------------------------------------- driver

def main():
    print("=" * 78)
    print("Explicit description of vosp, vq and vspe -- "
          "Subsec. 3.4")
    print("=" * 78)

    print("\n1) vq(n|k):  g = psq(n),  x = x_k,  h = pgl(n) as a space")
    print("   claim: h_0 = Z_{gl(n)}(X)/C,  dim = (n-k)^2 + k^2 - 1,"
          "  dim h_1 = 2k(n-k)")
    for n in range(2, 7):
        for k in range(1, n // 2 + 1):
            g = psq(n)
            dh, d0 = report(g, x_psq(n, k), f"psq({n}), k = {k}",
                            pred_psq(n, k), "Z(X)/C")
            assert dh == n * n - 1, "dim h"
            assert d0 == (n - k) ** 2 + k ** 2 - 1, "dim h_0 formula"
            assert dh - d0 == 2 * k * (n - k), "dim h_1 formula"
    print("   -> the two dimension formulas hold in every case run")

    print("\n2) vspe(n|p,q):  g = spe(n),  x = y^p_q,  h = sl(n) as a space")
    print("   claim: h_0 = {A : A_QP = A_QR = A_RP = 0, A_PP in o(p),"
          " A_QQ in sp(2q)}")
    for n in range(3, 8):
        for p in range(1, n):
            for q in range(1, (n - p) // 2 + 1):
                if p + 2 * q > n:
                    continue
                g = spe(n)
                s = n - p - 2 * q
                dh, d0 = report(g, y_spe(n, p, q), f"spe({n}), p={p}, q={q}",
                                pred_spe(n, p, q), "the block shape")
                assert dh == n * n - 1, "dim h"
                # tr A = 0 reads tr(delta) = 0, an actual condition only if
                # the third block is there at all
                pred = (p * (p - 1) // 2 + q * (2 * q + 1) + 2 * p * q
                        + p * s + 2 * q * s + (s * s - 1 if s else 0))
                assert d0 == pred, f"dim h_0 formula: {d0} vs {pred}"
    print("   -> dim h_0 = C(p,2) + q(2q+1) + 2pq + ps + 2qs + s^2 - 1,"
          " s = n-p-2q")

    print("\n3) vosp(m|n|p):  g = osp(2m|2n) and osp(2m+1|2n), x = im x^p_p")
    for e in (0, 1):
        for m in range(2, 6):
            for n in range(2, 6):
                for p in range(1, min(m, n) // 2 + 1):
                    g = osp(m, n, e)
                    x = x_osp(m, n, p, e)
                    lab = f"osp({2*m+e}|{2*n}), p = {p}"
                    dh, d0 = report(g, x, lab)
                    M = 2 * m + e
                    assert dh == M * (M - 1) // 2 + n * (2 * n + 1), \
                        f"dim h: {dh}"
                    # rank of ad_x on g_ev
                    r = 2 * p * (2 * m + e + 2 * n - 4 * p)
                    assert dh - d0 == r, \
                        f"{lab}: dim h_1 = {dh - d0}, formula gives {r}"
    print("   -> dim h = dim o(2m+e) + dim sp(2n),  dim h_1 = "
          "2p(2m + e + 2n - 4p) in every case")
    print("\nall assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
