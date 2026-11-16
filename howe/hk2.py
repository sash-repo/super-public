#!/usr/bin/env python3
"""The second alternative of the Problem on three connections (hyper-Kaehler section).

The Problem says, of the operators attached to
a~hyper-Kaehler manifold with a line bundle carrying connections nabla_a whose
curvature forms are hbar_a omega_a,

    X_a^+ = omega_a . ,   X_a^- = its adjoint,   a = 1, 2, 3,
    D_a^+ = d + hbar_a theta_a . ,   D_a^- = [X_a^-, D_a^+],

that the three-connection object is not osp(1|6), and that there are two candidates:

  1) the odd operators do not close, and one gets something larger than osp(1|4);
  2) "the third pair is a linear combination of the first two over the functions,
     and (D4.17) is right as printed while (D4.16) with three kernels is
     a consequence of its two conditions in the twisted setting though not in the
     untwisted one."

Everything is exact over Q, in the flat model H^m = R^{4m}, on
Omega = C[x_1..x_N] (x) Lambda(xhat_1..xhat_N), N = 4m, truncated by polynomial
degree; every identity is read off columns whose polynomial degree leaves the
truncation two units to spare, so that nothing compared is ever cut.

  E0.  iota_a := [X_a^-, theta_a . ] is ONE AND THE SAME operator for a = 1,2,3:
       the contraction with half the Euler vector field.  (Reason: theta_a is
       1/2 omega_a(E, -) and X_a^- convolves with omega_a^{-1}.)

  E1.  D_a^- = delta_a + hbar_a iota, where delta_a = [X_a^-, d]; and
       [delta_a, delta_b] = 0 = [iota, iota], so

           [D_a^-, D_b^-] = hbar_b [delta_a, iota] + hbar_a [iota, delta_b],

       which vanishes identically at hbar = 0.

  E2.  [delta_a, iota] = -X_a^-, whence

           [D_a^-, D_b^-] = -hbar_b X_a^- - hbar_a X_b^- .

       For a = b this is the square-root relation [D_a^-,D_a^-] = -2 hbar_a X_a^-.
       For a =/= b the right-hand side involves NO X_c^- with c =/= a, b.  So the
       bracket of the two odd operators of (D4.17) produces nothing beyond what
       their own squares produce, and X_3^- is NOT entailed.

  E3.  The literal wording of alternative 2 would force J_3 = f J_1 + g J_2
       pointwise; the three matrices are linearly independent.

  E4.  Kernels: the dimensions of ker D_1^- (^) ker D_2^- and of that intersected
       with ker X_3^-, computed on the safe part of the truncated model, for
       several values of (hbar_1, hbar_2).

  F.   The algebra itself.  Every operator here is quadratic in the generators
       x_i, d/dx_i, xhat_i, iota_i of the super Weyl algebra, so Q |-> [Q, -] is
       a faithful-modulo-constants representation of the algebra generated on the
       4N-dimensional span of those generators.  The closure is computed there,
       exactly, for two connections and for three.

Usage:  python3 hk2.py [m] [pdeg]
"""
import sys
from fractions import Fraction as F

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
try:
    from exactlin import nullspace as _xnull
except Exception:                                        # pragma: no cover
    _xnull = None

from hk import (quaternion_structures, check_quaternion_relations, Ext,
                mat_wedge_form, transpose, Twisted, tw_apply_d, tw_apply_theta,
                tw_apply_mat, rank)


# ------------------------------------------------------- operators as closures
# An operator is a pair (f, parity), f: dict -> dict on the truncated space.

def op_bracket(A, B):
    fa, pa = A
    fb, pb = B
    sgn = -1 if (pa and pb) else 1           # [A,B] = AB - (-1)^{pa pb} BA

    def f(v):
        u = fa(fb(v))
        w = fb(fa(v))
        out = dict(u)
        for k, c in w.items():
            out[k] = out.get(k, F(0)) - sgn * c
        return {k: c for k, c in out.items() if c}

    return (f, (pa + pb) % 2)


def op_lin(terms):
    par = terms[0][1][1]

    def f(v):
        out = {}
        for c, (g, _p) in terms:
            for k, val in g(v).items():
                out[k] = out.get(k, F(0)) + c * val
        return {k: c for k, c in out.items() if c}

    return (f, par)


def op_sparse(A, cols):
    """the operator's action on `cols`, as a sparse dict {(j, key): coeff}."""
    f = A[0]
    out = {}
    for j, b in enumerate(cols):
        for k, c in f({b: F(1)}).items():
            if c:
                out[(j, k)] = c
    return out


def sp_is_zero(s):
    return not s


def sp_solve(target, basis):
    """coefficients c with target = sum c_k basis[k]; and whether it is exact."""
    keys = set(target)
    for b in basis:
        keys |= set(b)
    keys = sorted(keys, key=repr)
    n = len(basis)
    rows = [[b.get(k, F(0)) for b in basis] + [target.get(k, F(0))] for k in keys]
    r, piv = 0, []
    for c in range(n):
        p = None
        for i in range(r, len(rows)):
            if rows[i][c]:
                p = i
                break
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        inv = F(1) / rows[r][c]
        rows[r] = [v * inv for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                q = rows[i][c]
                rows[i] = [a - q * b for a, b in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    co = [F(0)] * n
    for i, cc in enumerate(piv):
        co[cc] = rows[i][n]
    exact = all(target.get(k, F(0)) == sum(co[t] * basis[t].get(k, F(0))
                                           for t in range(n)) for k in keys)
    return co, exact


# ------------------------------------------------------------------- the model

def build(m, pdeg):
    N = 4 * m
    J = quaternion_structures(m)
    assert check_quaternion_relations(J), "quaternion relations fail"
    E = Ext(N)
    OM = [[[-J[a][i][j] for j in range(N)] for i in range(N)] for a in range(3)]
    Xp = [mat_wedge_form(E, OM[a]) for a in range(3)]
    Xm = [transpose(M) for M in Xp]
    T = Twisted(N, pdeg)

    d = (lambda v: tw_apply_d(T, v), 1)
    theta = [(lambda v, om=OM[a]: tw_apply_theta(T, om, v), 1) for a in range(3)]
    XM = [(lambda v, M=Xm[a]: tw_apply_mat(T, M, v), 0) for a in range(3)]
    XP = [(lambda v, M=Xp[a]: tw_apply_mat(T, M, v), 0) for a in range(3)]
    delta = [op_bracket(XM[a], d) for a in range(3)]
    iota = [op_bracket(XM[a], theta[a]) for a in range(3)]
    return dict(N=N, J=J, E=E, OM=OM, T=T, d=d, theta=theta, XM=XM, XP=XP,
                delta=delta, iota=iota)


def safe_columns(T, margin=2, maxform=None):
    out = [b for b in T.basis if sum(b[0]) <= T.pdeg - margin]
    if maxform is not None:
        out = [b for b in out if len(b[1]) <= maxform]
    return out


# ------------------------------------------------- the generators of super Weyl

def weyl_generators(B):
    """x_i, d/dx_i, xhat_i, iota_i as operators on the truncated space."""
    T, N = B["T"], B["N"]
    gens, names = [], []

    def mk_x(i):
        def f(v):
            out = {}
            for (xm, fm), c in v.items():
                nx = list(xm)
                nx[i] += 1
                if sum(nx) > T.pdeg:
                    continue
                out[(tuple(nx), fm)] = out.get((tuple(nx), fm), F(0)) + c
            return {k: c for k, c in out.items() if c}
        return (f, 0)

    def mk_dx(i):
        def f(v):
            out = {}
            for (xm, fm), c in v.items():
                if not xm[i]:
                    continue
                nx = list(xm)
                nx[i] -= 1
                k = (tuple(nx), fm)
                out[k] = out.get(k, F(0)) + c * xm[i]
            return {k: c for k, c in out.items() if c}
        return (f, 0)

    def mk_hx(i):
        def f(v):
            out = {}
            for (xm, fm), c in v.items():
                r = T.E.wedge_gen(i, fm)
                if r is None:
                    continue
                s, nf = r
                k = (xm, nf)
                out[k] = out.get(k, F(0)) + c * s
            return {k: c for k, c in out.items() if c}
        return (f, 1)

    def mk_io(i):
        def f(v):
            out = {}
            for (xm, fm), c in v.items():
                r = T.E.contract_gen(i, fm)
                if r is None:
                    continue
                s, nf = r
                k = (xm, nf)
                out[k] = out.get(k, F(0)) + c * s
            return {k: c for k, c in out.items() if c}
        return (f, 1)

    for i in range(N):
        gens.append(mk_x(i));  names.append(f"x{i}")
    for i in range(N):
        gens.append(mk_dx(i)); names.append(f"d{i}")
    for i in range(N):
        gens.append(mk_hx(i)); names.append(f"h{i}")
    for i in range(N):
        gens.append(mk_io(i)); names.append(f"i{i}")
    return gens, names


def rho(B, Q, gens, gvecs, cols):
    """the matrix of  u |-> [Q, u]  in the basis `gens` of the generators."""
    n = len(gens)
    M = [[F(0)] * n for _ in range(n)]
    for j, u in enumerate(gens):
        br = op_sparse(op_bracket(Q, u), cols)
        co, exact = sp_solve(br, gvecs)
        assert exact, "[Q,u] is not a combination of the generators"
        for i in range(n):
            M[i][j] = co[i]
    return M


def smatmul(A, Bm):
    n = len(A)
    Bt = list(zip(*Bm))
    return [[sum(A[i][t] * Bt[j][t] for t in range(n)) for j in range(n)]
            for i in range(n)]


def sbracket(A, pa, Bm, pb):
    P = smatmul(A, Bm)
    Q = smatmul(Bm, A)
    sgn = -1 if (pa and pb) else 1
    return [[P[i][j] - sgn * Q[i][j] for j in range(len(A))] for i in range(len(A))]


def closure(mats, pars, cap=200):
    basis, vecs, ps = [], [], []

    def add(M, p):
        v = [F(x) for row in M for x in row]
        if not any(v):
            return False
        if rank(vecs + [v]) > len(vecs):
            vecs.append(v)
            basis.append(M)
            ps.append(p)
            return True
        return False

    for M, p in zip(mats, pars):
        add(M, p)
    changed = True
    while changed and len(basis) < cap:
        changed = False
        cur = list(zip(basis, ps))
        for i in range(len(cur)):
            for j in range(i, len(cur)):
                Mi, pi = cur[i]
                Mj, pj = cur[j]
                if add(sbracket(Mi, pi, Mj, pj), (pi + pj) % 2):
                    changed = True
    return basis, ps


def _coord_reader(vecs):
    """pick n coordinates on which the n basis vectors are independent, invert the
    resulting n x n matrix once; then coords(v) is one matrix-vector product."""
    n = len(vecs)
    L = len(vecs[0])
    # Gauss on the n x L matrix whose ROWS are the basis vectors, to find pivots
    rows = [list(v) for v in vecs]
    piv, r = [], 0
    for c in range(L):
        p = None
        for i in range(r, n):
            if rows[i][c]:
                p = i
                break
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        inv = F(1) / rows[r][c]
        rows[r] = [v * inv for v in rows[r]]
        for i in range(n):
            if i != r and rows[i][c]:
                q = rows[i][c]
                rows[i] = [a - q * b for a, b in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
        if r == n:
            break
    assert r == n, "basis vectors are dependent"
    # A[i][k] = vecs[k][piv[i]];  coords solve A c = v|_piv
    A = [[vecs[k][piv[i]] for k in range(n)] for i in range(n)]
    # invert A
    aug = [A[i] + [F(1) if i == j else F(0) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = next(i for i in range(c, n) if aug[i][c])
        aug[c], aug[p] = aug[p], aug[c]
        inv = F(1) / aug[c][c]
        aug[c] = [v * inv for v in aug[c]]
        for i in range(n):
            if i != c and aug[i][c]:
                q = aug[i][c]
                aug[i] = [a - q * b for a, b in zip(aug[i], aug[c])]
    Ainv = [row[n:] for row in aug]

    def coords(v):
        sub = [v[piv[i]] for i in range(n)]
        return [sum(Ainv[k][i] * sub[i] for i in range(n)) for k in range(n)]

    return coords


def killing_rank_mats(basis, ps):
    """rank of the (super) Killing form of the matrix Lie superalgebra."""
    n = len(basis)
    vecs = [[F(x) for row in Bm for x in row] for Bm in basis]
    _rd = _coord_reader(vecs)

    def coords(M):
        return _rd([F(x) for row in M for x in row])

    ad = []
    for i in range(n):
        Mi = [[F(0)] * n for _ in range(n)]
        for j in range(n):
            cij = coords(sbracket(basis[i], ps[i], basis[j], ps[j]))
            for k in range(n):
                Mi[k][j] = cij[k]
        ad.append(Mi)
    K = [[F(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            P = smatmul(ad[i], ad[j])
            K[i][j] = sum(P[t][t] * (-1 if ps[t] else 1) for t in range(n))
    return rank(K)


def nullspace(M):
    if _xnull is not None:
        return _xnull([[F(v) for v in r] for r in M])
    rows = [[F(v) for v in r] for r in M]
    ncol = len(rows[0])
    piv, r = [], 0
    for c in range(ncol):
        p = None
        for i in range(r, len(rows)):
            if rows[i][c]:
                p = i
                break
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        inv = F(1) / rows[r][c]
        rows[r] = [v * inv for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                q = rows[i][c]
                rows[i] = [a - q * b for a, b in zip(rows[i], rows[r])]
        piv.append(c)
        r += 1
    free = [c for c in range(ncol) if c not in piv]
    out = []
    for fc in free:
        v = [F(0)] * ncol
        v[fc] = F(1)
        for i, pc in enumerate(piv):
            v[pc] = -rows[i][fc]
        out.append(v)
    return out


# ==================================================================== the runs

def run_E(m, pdeg, verbose=True):
    B = build(m, pdeg)
    T, N = B["T"], B["N"]
    cols = safe_columns(T, margin=2)
    delta, iota, XM = B["delta"], B["iota"], B["XM"]
    if verbose:
        print(f"=== m={m}  (N={N}, pdeg={pdeg}, truncated dim={len(T.basis)}, "
              f"columns compared={len(cols)})")

    sX = [op_sparse(XM[a], cols) for a in range(3)]
    sio = [op_sparse(iota[a], cols) for a in range(3)]

    # ---- E0: the three iota_a coincide
    same = all(sio[a] == sio[0] for a in range(3))
    if verbose:
        print(f"  [E0] iota_1 = iota_2 = iota_3 (contraction with E/2): {same}")

    # ---- E1
    dd = all(sp_is_zero(op_sparse(op_bracket(delta[a], delta[b]), cols))
             for a in range(3) for b in range(3))
    ii = all(sp_is_zero(op_sparse(op_bracket(iota[a], iota[b]), cols))
             for a in range(3) for b in range(3))
    if verbose:
        print(f"  [E1] [delta_a,delta_b]=0 for all a,b: {dd};   "
              f"[iota_a,iota_b]=0: {ii}")

    # ---- E2
    C, allin = {}, True
    for a in range(3):
        for b in range(3):
            co, ex = sp_solve(op_sparse(op_bracket(delta[a], iota[b]), cols), sX)
            C[(a, b)] = co
            allin = allin and ex
    if verbose:
        print(f"  [E2] every [delta_a,iota_b] is in Span(X_1^-,X_2^-,X_3^-): {allin}")
        for a in range(3):
            print(f"       [delta_{a+1}, iota] = "
                  + " + ".join(f"({C[(a,0)][k]}) X_{k+1}^-" for k in range(3)))
        print("       hence, with D_a^- = delta_a + hbar_a iota:")
        for (a, b) in ((0, 0), (0, 1), (0, 2), (1, 2)):
            terms = []
            for k in range(3):
                cb, ca = C[(a, b)][k], C[(b, a)][k]
                if cb or ca:
                    parts = ([f"{cb} hbar_{b+1}"] if cb else []) + \
                            ([f"{ca} hbar_{a+1}"] if ca else [])
                    terms.append("(" + " + ".join(parts) + f") X_{k+1}^-")
            print(f"          [D_{a+1}^-, D_{b+1}^-] = "
                  + (" + ".join(terms) if terms else "0"))

    # ---- E3
    Jv = [[F(x) for row in B["J"][a] for x in row] for a in range(3)]
    r12, r123 = rank(Jv[:2]), rank(Jv)
    if verbose:
        print(f"  [E3] rank(J_1,J_2)={r12}, rank(J_1,J_2,J_3)={r123}"
              f"  ->  J_3 = f J_1 + g J_2 pointwise: {r123 == r12}")

    # ---- E4: kernels, for several (hbar_1, hbar_2)
    if verbose:
        print("  [E4] on the safe part of the model "
              f"({len(cols)} columns), by (hbar_1,hbar_2):")
    def kerdim(ops):
        M = []
        for A in ops:
            f = A[0]
            imgs, keys = [], set()
            for b in cols:
                img = f({b: F(1)})
                imgs.append(img)
                keys |= set(img)
            for k in sorted(keys, key=repr):
                M.append([img.get(k, F(0)) for img in imgs])
        ns = nullspace(M) if M else [[F(1) if i == j else F(0)
                                      for i in range(len(cols))]
                                     for j in range(len(cols))]
        return ns

    # the untwisted contrast, on the same test space
    nsX = kerdim([XM[0], XM[1]])
    nsX3 = kerdim([XM[0], XM[1], XM[2]])
    if verbose:
        print(f"       untwisted: dim ker X_1^- (^) ker X_2^- = {len(nsX)}, "
              f"with ker X_3^- too = {len(nsX3)}"
              f"   -> inclusion: {len(nsX) == len(nsX3)}")
    for (h1, h2) in ((F(1), F(1)), (F(1), F(3)), (F(2), F(-5))):
        Dm = [op_lin([(F(1), delta[a]), (h, iota[a])])
              for a, h in ((0, h1), (1, h2))]
        ns = kerdim(Dm)
        n12 = len(ns)
        bad = 0
        for v in ns:
            vec = {b: v[j] for j, b in enumerate(cols) if v[j]}
            if XM[2][0](vec):
                bad += 1
        if verbose:
            print(f"       hbar=({h1},{h2}): dim ker D_1^- (^) ker D_2^- = {n12}, "
                  f"of which killed by X_3^- as well: {n12 - bad}"
                  f"   -> inclusion: {bad == 0}")
    return B, cols


def describe(basis, ps, label, extra=None, verbose=True):
    ev = sum(1 for p in ps if p == 0)
    od = sum(1 for p in ps if p == 1)
    kr = killing_rank_mats(basis, ps)
    if verbose:
        print(f"      {label}: sdim ({ev}|{od}), Killing rank {kr}"
              f" ({'nondegenerate' if kr == ev + od else 'degenerate'})"
              + (f"   {extra}" if extra else ""))
    return ev, od, kr


def even_part_report(basis, ps):
    """the even part: its dimension, the dimension of its derived algebra and of
    its center, and the rank of its own Killing form."""
    evb = [B for B, p in zip(basis, ps) if p == 0]
    n = len(evb)
    vecs = [[F(x) for row in Bm for x in row] for Bm in evb]

    def inspan(M, vs):
        v = [F(x) for row in M for x in row]
        return rank(vs + [v]) == len(vs)

    der = []
    dv = []
    for i in range(n):
        for j in range(i + 1, n):
            Cm = sbracket(evb[i], 0, evb[j], 0)
            v = [F(x) for row in Cm for x in row]
            if any(v) and rank(dv + [v]) > len(dv):
                dv.append(v)
                der.append(Cm)
    # center of the even part
    cen = 0
    rows = []
    for i in range(n):
        rr = []
        for j in range(n):
            rr.append(sbracket(evb[i], 0, evb[j], 0))
        rows.append(rr)
    # solve for combinations killing every generator
    big = []
    for j in range(n):
        for r in range(len(evb[0])):
            for c in range(len(evb[0])):
                big.append([rows[i][j][r][c] for i in range(n)])
    cen = n - rank([[big[t][i] for t in range(len(big))] for i in range(n)])
    return n, len(dv), cen


def run_F(B, cols, hbars=(F(1), F(2), F(3)), verbose=True):
    """the algebras generated, in the adjoint-on-generators model."""
    gens, names = weyl_generators(B)
    gcols = [b for b in cols if len(b[1]) <= 2]
    gvecs = [op_sparse(g, gcols) for g in gens]
    keys = sorted(set().union(*[set(v) for v in gvecs]), key=repr)
    Mg = [[v.get(k, F(0)) for v in gvecs] for k in keys]
    assert rank([[Mg[i][j] for i in range(len(keys))] for j in range(len(gens))]) \
        == len(gens), "the columns do not separate the generators"

    d, theta, XM, XP = B["d"], B["theta"], B["XM"], B["XP"]
    Dp, Dm = [], []
    for a in range(3):
        Dpa = op_lin([(F(1), d), (hbars[a], theta[a])])
        Dp.append(Dpa)
        Dm.append(op_bracket(XM[a], Dpa))

    Rp = [rho(B, Dp[a], gens, gvecs, gcols) for a in range(3)]
    Rm = [rho(B, Dm[a], gens, gvecs, gcols) for a in range(3)]
    RXm = [rho(B, XM[a], gens, gvecs, gcols) for a in range(3)]
    RXp = [rho(B, XP[a], gens, gvecs, gcols) for a in range(3)]

    if verbose:
        print(f"  [F] hbar = {tuple(str(h) for h in hbars)}")
        print("      --- calibration ---")
    # (a) Verbitsky
    bb, pp = closure(RXp + RXm, [0] * 6)
    describe(bb, pp, "{X_a^+, X_a^-}                 ", "expected sp(4): (10|0)", verbose)
    # (b) one connection, the Kaehler case
    bb, pp = closure([Rp[0], Rm[0]], [1, 1])
    describe(bb, pp, "{D_1^+, D_1^-}                 ", "expected osp(1|2): (3|2)", verbose)
    # (c) one twisted differential + the Verbitsky algebra  ([LKae], Thm on osp(2|4))
    bb, pp = closure([Rp[0]] + RXp + RXm, [1] + [0] * 6)
    describe(bb, pp, "{D_1^+, X_a^+, X_a^-}          ", "expected osp(2|4): (11|8)", verbose)

    if verbose:
        print("      --- the question ---")
    out = {}
    for k, lab in ((2, "{D_a^+, D_a^-: a=1,2}          "),
                   (3, "{D_a^+, D_a^-: a=1,2,3}        ")):
        mats = [m for a in range(k) for m in (Rp[a], Rm[a])]
        basis, ps = closure(mats, [1] * len(mats))
        vecs = [[F(x) for row in Bm for x in row] for Bm in basis]
        v3 = [F(x) for row in RXm[2] for x in row]
        inside = rank(vecs + [v3]) == len(vecs)
        ev, od, kr = describe(basis, ps, lab,
                              f"X_3^- inside: {inside}", verbose)
        n, nd, nc = even_part_report(basis, ps)
        if verbose:
            print(f"          even part: dim {n}, derived {nd}, center {nc}")
        out[k] = (ev, od, kr, inside)
    return out


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    pdeg = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print("Problem 7.3.1: what [D_1^-, D_2^-] is, and what the D's generate")
    print()
    B, cols = run_E(m, pdeg)
    print()
    run_F(B, cols, hbars=(F(1), F(2), F(3)))
    print()
    run_F(B, cols, hbars=(F(1), F(1), F(1)))


if __name__ == "__main__":
    main()
