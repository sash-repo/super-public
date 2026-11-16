#!/usr/bin/env python3
r"""Models of the exceptional simple vectorial Lie superalgebras."""
import sys
from fractions import Fraction
from itertools import combinations

from vfcore import Space                                            # noqa: E402
from excalg import (lam_basis, lam_mul, lam_der, vect0n_basis,      # noqa: E402
                    vect0n_act, vect0n_div, lin_field, prolong,
                    check_closed, span_index, echelon, pivots_of, in_span)
from vfcore import vec_of                                           # noqa: E402


def make_space(basis, par, wt):
    """basis: labels, par: dict label->parity, wt: dict label->weight tuple."""
    order = [b for b in basis if par[b] == 0] + [b for b in basis if par[b] == 1]
    nev = sum(1 for b in basis if par[b] == 0)
    nod = len(basis) - nev
    W = [tuple(-x for x in wt[b]) for b in order]      # weight of the coordinate
    S = Space(nev, nod, deg=[1] * len(order), wt=W)
    return S, order, {b: i for i, b in enumerate(order)}


def g0_from_action(S, pos, acts):
    """acts: list of dicts (c,b) -> coeff (matrices A with A(e_c) = sum A[c,b] e_b)."""
    bp = {a: (0 if a < S.nev else 1) for a in range(S.n)}
    out = []
    for A in acts:
        D = lin_field(S, A, bp)
        if D:
            out.append(D)
    return out


# --------------------------------------------------------------- vle(4|3;0)
def build_vle(top=2):
    n = 3
    labels = [S for S in lam_basis(n) if len(S) > 0]        # Lambda(3)/C.1
    par = {S: (len(S) + 1) % 2 for S in labels}             # Pi
    wt = {S: tuple(1 if i in S else 0 for i in range(n)) for S in labels}
    Sp, order, pos = make_space(labels, par, wt)
    acts = []
    for (T, i) in vect0n_basis(n):
        A = {}
        for Sset in labels:
            for S2, c in vect0n_act(n, T, i, Sset).items():
                if len(S2) == 0:
                    continue                                # killed in Lambda/C.1
                A[(pos[Sset], pos[S2])] = A.get((pos[Sset], pos[S2]), 0) + c
        if A:
            acts.append(A)
    acts.append({(a, a): Fraction(1) for a in range(Sp.n)})  # the scalar
    g0 = g0_from_action(Sp, pos, acts)
    gm = {-1: [{(a, Sp.one()): Fraction(1)} for a in range(Sp.n)]}
    return Sp, gm, g0, top


# ----------------------------------------------------------- kas(1|6; 3eta)
def build_kas(top=2):
    """(Vol_0(0|3), c vect(0|3))_*, with vvol taken even so that sdim g_- = (4|3)."""
    n = 3
    labels = [S for S in lam_basis(n) if len(S) < n]        # Vol_0: no top form
    par = {S: len(S) % 2 for S in labels}
    wt = {S: tuple(1 if i in S else 0 for i in range(n)) for S in labels}
    Sp, order, pos = make_space(labels, par, wt)
    acts = []
    for (T, i) in vect0n_basis(n):
        A = {}
        pD = (len(T) + 1) % 2
        for Sset in labels:
            # D(f vvol) = (D(f) + (-1)^{p(f)p(D)} f Div D) vvol
            for S2, c in vect0n_act(n, T, i, Sset).items():
                if len(S2) == n:
                    continue
                A[(pos[Sset], pos[S2])] = A.get((pos[Sset], pos[S2]), 0) + c
            sgn = -1 if (len(Sset) % 2) * pD else 1
            for T1, c in vect0n_div(n, T, i).items():
                pr = lam_mul(Sset, T1)
                if pr is None:
                    continue
                s, S2 = pr
                if len(S2) == n:
                    continue
                A[(pos[Sset], pos[S2])] = A.get((pos[Sset], pos[S2]), 0) + sgn * s * c
        A = {k: v for k, v in A.items() if v}
        if A:
            acts.append(A)
    acts.append({(a, a): Fraction(1) for a in range(Sp.n)})
    g0 = g0_from_action(Sp, pos, acts)
    gm = {-1: [{(a, Sp.one()): Fraction(1)} for a in range(Sp.n)]}
    return Sp, gm, g0, top


# ------------------------------------------------------------- vas(4|4)
def build_vas(top=2):
    """(spin, as)_*, as embedded in gl(4|4) by the spinor representation T_1."""
    n = 4
    # basis of spin = C^{4|4}: e_0..e_3 even, f_0..f_3 odd
    labels = [('e', i) for i in range(n)] + [('f', i) for i in range(n)]
    # the parity is that of Pi(C^{4|4}): with the other one the prolong is 0,
    # whereas vas(4|4) is infinite-dimensional.
    par = {('e', i): 1 for i in range(n)}
    par.update({('f', i): 0 for i in range(n)})
    # weights: e_i has eps_i, f_j has -eps_j  (the torus is the diagonal of sl(4))
    # the torus is the traceless diagonal of sl(4) subset as, so the weights
    # live in Z^4 modulo (1,1,1,1); project by successive differences.
    def pi(w):
        return tuple(w[k] - w[k + 1] for k in range(n - 1))
    wt = {('e', i): pi([1 if k == i else 0 for k in range(n)]) for i in range(n)}
    wt.update({('f', i): pi([-1 if k == i else 0 for k in range(n)]) for i in range(n)})
    Sp, order, pos = make_space(labels, par, wt)

    def tilde(C):
        """C -> Ctilde on skew 4x4 matrices: C_{ij} := E_ij - E_ji, Ctilde_{ij} = C_{kl}."""
        out = [[Fraction(0)] * n for _ in range(n)]
        pairs = {(0, 1): (2, 3), (2, 3): (0, 1), (0, 2): (3, 1), (3, 1): (0, 2),
                 (0, 3): (1, 2), (1, 2): (0, 3)}
        for (i, j), (k, l) in pairs.items():
            out[k][l] += C[i][j]
            out[l][k] -= C[i][j]
        return out

    def mat_to_A(M):
        """M acts on V by v -> Mv;  A[c,b] = M[b][c]."""
        A = {}
        for b in range(2 * n):
            for c in range(2 * n):
                if M[b][c]:
                    A[(pos[labels[c]], pos[labels[b]])] = Fraction(M[b][c])
        return A

    acts = []
    # even part: A in sl(4) -> [[A,0],[0,-A^t]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            M = [[Fraction(0)] * (2 * n) for _ in range(2 * n)]
            M[i][j] = Fraction(1)
            M[n + j][n + i] = Fraction(-1)
            acts.append(mat_to_A(M))
    for i in range(n - 1):
        M = [[Fraction(0)] * (2 * n) for _ in range(2 * n)]
        M[i][i] = Fraction(1)
        M[i + 1][i + 1] = Fraction(-1)
        M[n + i][n + i] = Fraction(-1)
        M[n + i + 1][n + i + 1] = Fraction(1)
        acts.append(mat_to_A(M))
    # B symmetric (the block spe(4)_1 = S^2 V)
    for i in range(n):
        for j in range(i, n):
            M = [[Fraction(0)] * (2 * n) for _ in range(2 * n)]
            M[i][n + j] = Fraction(1)
            M[j][n + i] += Fraction(1)
            acts.append(mat_to_A(M))
    # C skew (the block spe(4)_{-1} = Lambda^2 V^*), twisted by -Ctilde
    for i in range(n):
        for j in range(i + 1, n):
            C = [[Fraction(0)] * n for _ in range(n)]
            C[i][j] = Fraction(1)
            C[j][i] = Fraction(-1)
            Ct = tilde(C)
            M = [[Fraction(0)] * (2 * n) for _ in range(2 * n)]
            for a in range(n):
                for b in range(n):
                    if C[a][b]:
                        M[n + a][b] = C[a][b]
                    if Ct[a][b]:
                        M[a][n + b] = -Ct[a][b]
            acts.append(mat_to_A(M))
    # the centre: a * 1_{4|4}
    acts.append({(a, a): Fraction(1) for a in range(Sp.n)})
    g0 = g0_from_action(Sp, pos, acts)
    gm = {-1: [{(a, Sp.one()): Fraction(1)} for a in range(Sp.n)]}
    return Sp, gm, g0, top


# --------------------------------------------------------------- mb(4|5;0)
def build_mb(top=2):
    """(ba(4), c vect(0|3))^m_* inside m(4|5).

    Coordinates of m(4): q_0..q_3 (even, deg 1), xi_0..xi_3 (odd, deg 1),
    tau (odd, deg 2).  M_f is the pericontact vector field.
    g_{-1} = Vol(0|3)^{1/2} = Lambda(3) (x) vvol^{1/2}, sdim (4|4);
    the identification with span(M_{q_i}, M_{xi_i}) is fixed below.
    """
    nev, nod = 4, 5                 # q_0..q_3 ; xi_0..xi_3, tau
    TAU = 4                         # local odd index of tau
    deg = [1] * 4 + [1] * 4 + [2]
    # torus weights: the three t_j of vect(0|3) act on g_{-1} = Lambda(3);
    # filled in after the identification below.
    Sp = Space(nev, nod, deg=deg, wt=[(0, 0, 0)] * 9)
    return Sp
