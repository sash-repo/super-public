#!/usr/bin/env python3
"""[BLO] = Ben Fraj, Laraiedh, Omri, "Supertransvectants, cohomology, and
deformations", J. Math. Phys. 54 (2013) 023501, in ITS OWN conventions (their
Sec. II), so that the comparison with Section 3 of the paper is independent
of any dictionary between the two normalizations.

Their conventions, verbatim:

    coordinates      x, theta_1, ..., theta_N,   alpha_N = dx + sum theta_i d theta_i
    eta_i            = d/d theta_i - theta_i d/dx,      eta_i^2 = -d_x
    contact fields   X_F = F d_x - (1/2) sum_i (-1)^{p(F)} eta_i(F) eta_i
    bracket          {F,G} = F G' - F' G - (1/2)(-1)^{p(F)} sum_i eta_i(F) eta_i(G)
                     [X_F, X_G] = X_{F,G}
    densities        F^N_lam = { F alpha_N^lam },  L^lam_{X_F} = X_F + lam F'
    osp(N|2)         = < X_1, X_x, X_{x^2}, X_{theta_i}, X_{x theta_i},
                          X_{theta_i theta_j} >

A bilinear operator T: F_tau (x) F_lam --> F_mu is invariant iff for all F

    L^mu_{X_F}(T(u,v)) = (-1)^{p(F)p(T)} [ T(L^tau_{X_F} u, v)
                                           + (-1)^{p(F)p(u)} T(u, L^lam_{X_F} v) ].

Monomials are the pairs (a, mask) = x^a theta^mask of superN.py.
"""
from fractions import Fraction as Fr
from itertools import combinations

from superN import (add, smul, mul, mono, parity, par_mon, d_t, d_odd,
                    popcount, clean, dtn, compose)

d_x = d_t                                   # their x is our t


# --------------------------------------------------------------- the setting

class BLO:
    def __init__(self, N):
        self.N = N
        self.d = [d_odd(i) for i in range(N)]

    # eta_i = d_i - theta_i d_x
    def eta(self, i):
        th = mono(0, 1 << i)

        def r(u):
            return add(self.d[i](u), smul(-1, mul(th, d_x(u))))
        return r

    def etaword(self, subset, k=0):
        """d_x^k eta_{i1} ... eta_{ir}, subset = (i1 < ... < ir)."""
        return compose(*[self.eta(i) for i in subset], dtn(k))

    # X_F = F d_x - (1/2) sum (-1)^{p(F)} eta_i(F) eta_i
    def X(self, F):
        pF = parity(F)
        s = Fr(-1, 2) * (-1 if pF else 1)
        eF = [self.eta(i)(F) for i in range(self.N)]

        def r(u):
            out = mul(F, d_x(u))
            for i in range(self.N):
                if eF[i]:
                    out = add(out, smul(s, mul(eF[i], self.eta(i)(u))))
            return out
        return r

    # {F,G} = F G' - F' G - (1/2)(-1)^{p(F)} sum eta_i(F) eta_i(G)
    def kb(self, F, G):
        pF = parity(F)
        s = Fr(-1, 2) * (-1 if pF else 1)
        out = add(mul(F, d_x(G)), smul(-1, mul(d_x(F), G)))
        for i in range(self.N):
            out = add(out, smul(s, mul(self.eta(i)(F), self.eta(i)(G))))
        return out

    # L^lam_{X_F} = X_F + lam F'
    def L(self, F, lam):
        XF = self.X(F)
        c = smul(lam, d_x(F))

        def r(u):
            return add(XF(u), mul(c, u))
        return r

    def osp_gens(self):
        """the 12 (for N=3) generating functions of osp(N|2) subset K(N)."""
        N = self.N
        out = [mono(0, 0), mono(1, 0), mono(2, 0)]                # 1, x, x^2
        out += [mono(0, 1 << i) for i in range(N)]                # theta_i
        out += [mono(1, 1 << i) for i in range(N)]                # x theta_i
        out += [mono(0, (1 << i) | (1 << j))
                for i in range(N) for j in range(i + 1, N)]       # theta_i theta_j
        return out

    def monos(self, tmax):
        return [(a, m) for a in range(tmax + 1) for m in range(1 << self.N)]


# ------------------------------------------------------------- the ansatz

def terms_of_order(N, ordr):
    """(a, A, b, B) with 2a+|A|+2b+|B| = ordr;  T = d_x^a eta_A (x) d_x^b eta_B."""
    subsets = [()] + [c for r in range(1, N + 1)
                      for c in combinations(range(N), r)]
    out = []
    for A in subsets:
        for B in subsets:
            rest = ordr - len(A) - len(B)
            if rest < 0 or rest % 2:
                continue
            for a in range(rest // 2 + 1):
                out.append((a, A, rest // 2 - a, B))
    return out


def label(t, names=None):
    a, A, b, B = t
    def one(k, S):
        s = "".join(f"e{i+1}" for i in S)
        if k:
            s = (f"dx^{k}" if k > 1 else "dx") + s
        return s or "1"
    return f"{one(a,A)}(x){one(b,B)}"


class Classifier:
    """dim of the space of osp(N|2)-invariant bilinear operators of a given
    order 2k, with the constant-coefficient ansatz (justified by invariance
    under X_1 and X_{theta_i}, verified in check_neg())."""

    def __init__(self, alg, tau, lam, gens=None):
        self.alg, self.tau, self.lam = alg, tau, lam
        self.gens = gens if gens is not None else alg.osp_gens()

    def rows(self, ordr, tmax=None):
        alg = self.alg
        ts = terms_of_order(alg.N, ordr)
        pT = ordr % 2
        mu = self.tau + self.lam + Fr(ordr, 2)
        tmax = tmax if tmax is not None else ordr // 2 + 2
        MS = alg.monos(tmax)
        Ps = [alg.etaword(A, a) for (a, A, b, B) in ts]
        Qs = [alg.etaword(B, b) for (a, A, b, B) in ts]
        pQ = [len(B) % 2 for (a, A, b, B) in ts]

        rows = {}
        for gi, f in enumerate(self.gens):
            pf = parity(f)
            sT = -1 if (pf and pT) else 1
            L1, L2 = alg.L(f, self.tau), alg.L(f, self.lam)
            Lt = alg.L(f, mu)
            Pu = [{m: Ps[i](mono(*m)) for m in MS} for i in range(len(ts))]
            Qv = [{m: Qs[i](mono(*m)) for m in MS} for i in range(len(ts))]
            L1u = {m: L1(mono(*m)) for m in MS}
            L2v = {m: L2(mono(*m)) for m in MS}
            PLu = [{m: Ps[i](L1u[m]) for m in MS} for i in range(len(ts))]
            QLv = [{m: Qs[i](L2v[m]) for m in MS} for i in range(len(ts))]
            for m1 in MS:
                pu = par_mon(m1)
                s2 = -1 if (pf and pu) else 1
                pLu = (pu + pf) % 2
                for m2 in MS:
                    for i in range(len(ts)):
                        sgn = -1 if (pQ[i] and pu) else 1
                        sgn1 = -1 if (pQ[i] and pLu) else 1
                        Buv = smul(sgn, mul(Pu[i][m1], Qv[i][m2]))
                        BLuv = smul(sgn1, mul(PLu[i][m1], Qv[i][m2]))
                        BuLv = smul(sgn, mul(Pu[i][m1], QLv[i][m2]))
                        res = add(Lt(Buv),
                                  smul(-sT, add(BLuv, smul(s2, BuLv))))
                        for k, val in res.items():
                            key = (gi, m1, m2, k)
                            rows.setdefault(key, {})[i] = \
                                rows.setdefault(key, {}).get(i, 0) + val
        return ts, rows

    def solve(self, ordr, tmax=None):
        ts, rows = self.rows(ordr, tmax)
        mat = [[r.get(i, 0) for i in range(len(ts))] for r in rows.values()]
        mat = [row for row in mat if any(x != 0 for x in row)]
        return ts, nullspace(mat, len(ts))


def nullspace(mat, ncols):
    from sympy import expand, cancel, S
    symbolic = any(not isinstance(x, (int, Fr)) for row in mat for x in row)
    if symbolic:
        M = [[S(x) for x in row] for row in mat]
        isz = lambda x: expand(x) == 0
        red = lambda x: cancel(expand(x))
    else:
        M = [[Fr(x) for x in row] for row in mat]
        isz = lambda x: x == 0
        red = lambda x: x
    pivots, r = [], 0
    for c in range(ncols):
        piv = next((rr for rr in range(r, len(M)) if not isz(M[rr][c])), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [red(x / pv) for x in M[r]]
        for rr in range(len(M)):
            if rr != r and not isz(M[rr][c]):
                f = M[rr][c]
                M[rr] = [red(a - f * b) for a, b in zip(M[rr], M[r])]
        pivots.append(c)
        r += 1
        if r == len(M):
            break
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for fc in free:
        v = [0] * ncols
        v[fc] = 1
        for i, pc in enumerate(pivots):
            v[pc] = -M[i][fc]
        basis.append(v)
    return basis


# ------------------------------------------- invariance test for a given T

def invariance_defect(alg, T, pT, tau, lam, tmax=3, gens=None):
    """T: a function (u,v) -> superfunction, of parity pT.  Returns the list of
    (generator index, u, v) where the invariance condition fails."""
    mu = tau + lam + Fr(0)      # unused; mu is passed through T's own order
    raise NotImplementedError


def check_invariance(alg, T, ordr, tau, lam, tmax=3, gens=None):
    """T(u,v) a bilinear operator of order `ordr` (parity ordr % 2) taking
    F_tau (x) F_lam --> F_{tau+lam+ordr/2}.  Returns the list of failures."""
    gens = gens if gens is not None else alg.osp_gens()
    pT = ordr % 2
    mu = tau + lam + Fr(ordr, 2)
    MS = alg.monos(tmax)
    bad = []
    for gi, f in enumerate(gens):
        pf = parity(f)
        sT = -1 if (pf and pT) else 1
        L1, L2, Lt = alg.L(f, tau), alg.L(f, lam), alg.L(f, mu)
        for m1 in MS:
            u = mono(*m1)
            pu = par_mon(m1)
            s2 = -1 if (pf and pu) else 1
            for m2 in MS:
                v = mono(*m2)
                res = add(Lt(T(u, v)),
                          smul(-sT, add(T(L1(u), v), smul(s2, T(u, L2(v))))))
                if clean(res):
                    bad.append((gi, m1, m2, res))
    return bad


# ------------------------------------------ [BLO]'s own explicit operators

def binom(x, i):
    """(x choose i) = x(x-1)...(x-i+1)/i!, x a number or a sympy symbol."""
    if i < 0:
        return 0
    num = 1
    for r in range(i):
        num = num * (x - r)
    from math import factorial
    return num / factorial(i) if not isinstance(num, (int, Fr)) \
        else Fr(num, factorial(i))


def Gamma(tau, lam, i, j, k):
    """(3.4): Gamma^{tau,lam}_{i,j,k} = (-1)^j C(2tau+[k], j) C(2lam+[k], i)."""
    kk = int(k) if k >= 0 else -int(-k)          # [k] = integer part, k >= 0
    kk = int(k // 1)
    return (-1) ** j * binom(2 * tau + kk, j) * binom(2 * lam + kk, i)


def Lam(i, j):
    """(4.2): Lambda^i_j = (j+2)(j+1)(i+1)."""
    return (j + 2) * (j + 1) * (i + 1)


def J_N1(alg, tau, lam, k):
    """(3.2)-(3.3): the osp(1|2) supertransvectants, N = 1."""
    e1 = alg.eta(0)
    kint = int(k)
    if k == int(k):                                     # even, k in N
        def T(u, v):
            pu = parity(u)
            s = -1 if pu else 1
            out = {}
            for i in range(kint):
                j = kint - 1 - i
                out = add(out, smul(s * Gamma(tau, lam, i, j, kint - 1),
                                    mul(e1(dtn(i)(u)), e1(dtn(j)(v)))))
            for i in range(kint + 1):
                j = kint - i
                out = add(out, smul(-Gamma(tau, lam, i, j, kint - 1),
                                    mul(dtn(i)(u), dtn(j)(v))))
            return out
    else:                                               # odd, k in N + 1/2
        kb = int(k // 1)                                # [k]
        def T(u, v):
            pu = parity(u)
            s = -1 if pu else 1
            out = {}
            for i in range(kb + 1):
                j = kb - i
                g = Gamma(tau, lam, i, j, kb)
                out = add(out,
                          smul(s * g * (2 * tau + kb - j),
                               mul(dtn(i)(u), e1(dtn(j)(v)))),
                          smul(-g * (2 * lam + kb - i),
                               mul(e1(dtn(i)(u)), dtn(j)(v))))
            return out
    return T


def J_N3(alg, tau, lam, k):
    """(4.1)-(4.2): [BLO]'s osp(3|2)-invariant operators, N = 3."""
    e = [alg.eta(i) for i in range(3)]
    e12 = lambda u: e[0](e[1](u))
    e123 = lambda u: e[0](e[1](e[2](u)))
    pairs = [(0, 1), (0, 2), (1, 2)]                    # 1<=p<l<=3, 0-based

    if k == int(k):                                     # (4.2), k in N
        kk = int(k)
        def T(u, v):
            pu = parity(u)
            s = -1 if pu else 1
            out = {}
            for i in range(kk + 1):                     # i+j=k
                j = kk - i
                out = add(out, smul(-Gamma(tau, lam, i, j, kk - 1),
                                    mul(dtn(i)(u), dtn(j)(v))))
            for p in range(3):                          # i+j=k-1
                for i in range(kk):
                    j = kk - 1 - i
                    out = add(out, smul(-s * Gamma(tau, lam, i, j, kk - 1),
                                        mul(e[p](dtn(i)(u)), e[p](dtn(j)(v)))))
            for (p, l) in pairs:                        # i+j=k-2
                for i in range(max(kk - 1, 0)):
                    j = kk - 2 - i
                    if j < 0:
                        continue
                    ep = lambda w: e[p](e[l](w))
                    out = add(out, smul(-Gamma(tau, lam, i, j, kk - 1),
                                        mul(ep(dtn(i)(u)), ep(dtn(j)(v)))))
            for i in range(max(kk - 2, 0)):             # i+j=k-3
                j = kk - 3 - i
                if j < 0:
                    continue
                out = add(out, smul(s * Gamma(tau, lam, i, j, kk - 1),
                                    mul(e123(dtn(i)(u)), e123(dtn(j)(v)))))
            return out
    else:                                               # (4.1), 2k in N\{0,1}
        kb = int(k // 1)                                # [k]
        def T(u, v):
            pu = parity(u)
            s = -1 if pu else 1
            out = {}
            for (p, l) in pairs:
                q = 3 - p - l                           # the third index
                for i in range(kb):
                    j = kb - 1 - i
                    if j < 0:
                        continue
                    # - Gamma_{i+1,j+2,k+1} Lam^i_j eta_q(F^(i)) eta_p eta_l(G^(j))
                    out = add(out, smul(
                        -Gamma(tau, lam, i + 1, j + 2, kb + 1) * Lam(i, j),
                        mul(e[q](dtn(i)(u)), e[p](e[l](dtn(j)(v))))))
                    # - (-1)^p(F) Gamma_{i+2,j+1,k+1} Lam^j_i eta_p eta_l(F^(i)) eta_q(G^(j))
                    out = add(out, smul(
                        -s * Gamma(tau, lam, i + 2, j + 1, kb + 1) * Lam(j, i),
                        mul(e[p](e[l](dtn(i)(u))), e[q](dtn(j)(v)))))
            for i in range(kb):
                j = kb - 1 - i
                if j < 0:
                    continue
                out = add(out, smul(
                    s * Gamma(tau, lam, i, j + 3, kb + 1) * Lam(j, j + 1),
                    mul(dtn(i)(u), e123(dtn(j)(v)))))
                out = add(out, smul(
                    Gamma(tau, lam, i + 3, j, kb + 1) * Lam(i, i + 1),
                    mul(e123(dtn(i)(u)), dtn(j)(v))))
            return out
    return T


def J_N3_signs(alg, tau, lam, k, eps=(-1, 1, -1, 1)):
    """[BLO]'s (4.2) (even k) with the sign of each of the four groups (those
    with r = 0,1,2,3 eta's on each argument) replaced by eps[r]: which sign
    pattern, if any, makes their formula invariant?"""
    e = [alg.eta(i) for i in range(3)]
    e123 = lambda u: e[0](e[1](e[2](u)))
    pairs = [(0, 1), (0, 2), (1, 2)]
    kk = int(k)

    def T(u, v):
        s = -1 if parity(u) else 1
        out = {}
        for i in range(kk + 1):
            j = kk - i
            out = add(out, smul(eps[0] * Gamma(tau, lam, i, j, kk - 1),
                                mul(dtn(i)(u), dtn(j)(v))))
        for p in range(3):
            for i in range(kk):
                j = kk - 1 - i
                out = add(out, smul(eps[1] * s * Gamma(tau, lam, i, j, kk - 1),
                                    mul(e[p](dtn(i)(u)), e[p](dtn(j)(v)))))
        for (p, l) in pairs:
            for i in range(kk - 1):
                j = kk - 2 - i
                ep = lambda w: e[p](e[l](w))
                out = add(out, smul(eps[2] * Gamma(tau, lam, i, j, kk - 1),
                                    mul(ep(dtn(i)(u)), ep(dtn(j)(v)))))
        for i in range(kk - 2):
            j = kk - 3 - i
            out = add(out, smul(eps[3] * s * Gamma(tau, lam, i, j, kk - 1),
                                mul(e123(dtn(i)(u)), e123(dtn(j)(v)))))
        return out
    return T
