#!/usr/bin/env python3
"""
n4vac.py -- the hunt for the exact N=4 vacuum superfield of (IK4).



WHAT WORKS.  The N=2 vacuum is exact and verified (n2noeth-vac.py):
    E  = 2 (m wt_v )^{-1} (1 - i rho_- eta_+ / wt_v),
    Eb = 2 (m wt_vb)^{-1} (1 - i eta_- rho_+ / wt_vb).
Per [IvKr4] (20)-(21) it embeds into N=4: this file builds the EMBEDDED BASE
q_base -- the unique-up-to-completion constrained q whose projection pi(q)
(second planes' thetas to zero) equals diag(E, Eb) -- exactly, by one linear
solve (build_embedded_base).

WHAT IS OPEN.  The completion of q_base to an exact solution of E1+E2.  Exact
Newton over every window tried so far reports an inconsistent E1-sector of
theta^4-terms; the two prepared routes:
  (a) ungraded windows (m-powers freed) -- expensive nullspace, in progress;
  (b) deterministic level-by-level reconstruction: on shell all eight odd
      derivatives of q are rank one (the (9') relations forced by E2's
      integrability), so the theta-expansion of q is determined recursively
      from the component data q_0 = Liouville, fermions = 0,
      C_2 = 4im Sp(q_0 bar q_0).

Usage: import from a driver; see build_embedded_base() and n4noeth.newton_m.
"""

import itertools
import pickle

from fractions import Fraction as Q

import n4noeth as N
from n4noeth import (mono, add, sub, smul, mul, inv_scalar, mzero, madd,
                     msmul, cDm, bDm, cDp, bDp, M_MONO, C, MI, PL,
                     check_constraints, residuals, mid)

SECOND = {3, 4, 7, 8}


def pi(f):
    """Set the second planes' thetas to zero."""
    return {k: v for k, v in f.items() if not (set(k[3]) & SECOND)}


def n2_structures():
    """The four scalar structures of the exact N=2 vacuum, on planes 1."""
    WV = add(mono(0, 1, 0), mono(0, 0, 0, (1, 2), Q(1, 2)),
             mono(0, 0, 0, (5, 6), Q(-1, 2)))
    WB = add(mono(0, 1, 0), mono(0, 0, 0, (1, 2), Q(-1, 2)),
             mono(0, 0, 0, (5, 6), Q(1, 2)))
    IV = inv_scalar(mul(M_MONO, WV), mono(0, 1, 1))
    IV2 = inv_scalar(WV, mono(0, 1, 0))
    IB = inv_scalar(mul(M_MONO, WB), mono(0, 1, 1))
    IB2 = inv_scalar(WB, mono(0, 1, 0))
    RH = mono(0, 0, 0, (2, 5))
    HR = mono(0, 0, 0, (1, 6))
    return (IV, mul(IV, mul(RH, IV2)), IB, mul(IB, mul(HR, IB2)))


def graded_monos(zlo, zhi, wlo, whi, mfree=None):
    """Grade-0 window monomials; nm tied to nz+nw+|S|/2 unless mfree is a
    range, in which case the m-powers run over it independently."""
    out = []
    for r in (0, 2, 4, 6, 8):
        for S in itertools.combinations(MI + PL, r):
            for a in range(zlo, zhi + 1):
                for b in range(wlo, whi + 1):
                    if mfree is None:
                        out.append(mono(a, b, a + b + r // 2, S))
                    else:
                        for c in range(mfree[0], mfree[1] + 1):
                            out.append(mono(a, b, c, S))
    return out


def constrained_basis(ms):
    """The pure C+- constrained matrices over the given entry monomials."""
    def comps(q):
        L = []
        for g in (0, 1):
            u = (q[g][0], q[g][1])
            L += [cDm(1, u[0]), cDm(2, u[1]),
                  add(cDm(1, u[1]), cDm(2, u[0]))]
        for b in (0, 1):
            v = (q[0][b], q[1][b])
            L += [bDp(1, v[0]), bDp(2, v[1]),
                  add(bDp(1, v[1]), bDp(2, v[0]))]
        return L
    cols, where = [], []
    for g in (0, 1):
        for b in (0, 1):
            for mo in ms:
                q = mzero()
                q[g][b] = mo
                col = {}
                for i, f in enumerate(comps(q)):
                    for k, c in f.items():
                        col[(i, k)] = c
                cols.append(col)
                where.append((g, b, mo))
    basis = []
    for v in N.sparse_null(cols):
        q = mzero()
        for j, c in enumerate(v):
            if c:
                g, b, mo = where[j]
                q[g][b] = add(q[g][b], smul(c, mo))
        basis.append(q)
    return basis


def build_embedded_base(B):
    """q with C+-(q) = 0 and pi(q) = diag(E, Eb) exactly, plus the kernel of
    the joint linear system (the completion directions with pi = 0)."""
    E1s, E2s, B1s, B2s = n2_structures()
    nB = len(B)
    cols = []
    for M in B:
        col = {}
        for i in (0, 1):
            for jj in (0, 1):
                for k, v in pi(M[i][jj]).items():
                    col[(2 * i + jj, k)] = v
        cols.append(col)
    for (slot, S_) in ((0, E1s), (0, E2s), (3, B1s), (3, B2s)):
        col = {}
        for k, v in S_.items():
            col[(slot, k)] = -v
        cols.append(col)
    null = N.sparse_null(cols)
    sc_vecs = [v for v in null if any(v[nB:])]
    want = [C(2), C(0, -2), C(2), C(0, -2)]
    A = [[sc_vecs[j][nB + r] for r in range(4)] for j in range(len(sc_vecs))]
    x = N.csolve(A, want)
    if x is None:
        raise RuntimeError("no combination matches the N=2 vacuum scalars")
    base = [C() for _ in range(nB + 4)]
    for j, c in enumerate(x):
        if c:
            for i in range(nB + 4):
                base[i] = base[i] + c * sc_vecs[j][i]
    q = mzero()
    for j in range(nB):
        if base[j]:
            q = madd(q, msmul(base[j], B[j]))
    kern = []
    for v in null:
        if not any(v[nB:]):
            qq = mzero()
            for j in range(nB):
                if v[j]:
                    qq = madd(qq, msmul(v[j], B[j]))
            kern.append(qq)
    return q, kern
