#!/usr/bin/env python3
"""Recomputation of the degree-1 part of [GLS, Sec.10] for kas and k(1|6) on
the fibres S^m_0 -- the fibres in which the families 1a) and 1a^dagger) live.

Subsection 6.7 of the paper says: the family 1a) of [GLS, Sec.10] is not closed under the
formal adjoint, the missing family 1a^dagger) being

    1a^dagger)   lam = (k, k-4, l, l)  -->  lam + (-1,-1,0,0),

of which the divergence is the member (k, l) = (5, 0).  For l = 0 the fibres
of both families are the traceless symmetric powers S^m_0 (symtype.py):

    1a)          T(S^m_0)_{lam}      --> T(S^{m+1}_0)_{lam + 2}   (the
                 "symmetrized gradient"; m = 0 gives grad),
    1a^dagger)   T(S^{m+1}_0)_{-4}   --> T(S^m_0)_{-4}            (the
                 "divergence"; m = 0 gives div).

Here we compute both directions from scratch, for m = 0, 1, 2, with the
engine of kasV.py, over kas and over k(1|6):

  (1) the K_t-line lam2 = lam1 + c, then the full invariance problem along it
      for lam1 in [-8, 8] with step 1/2 (gls.scan);
  (2) every operator found is re-verified by the independent routine of
      glscheck.py -- the module action only, all generating functions, all
      test monomials -- and is checked to FAIL at the neighbouring weights;
  (3) every operator found is looked up in [GLS, Sec.10] with dagger.in_list.

Usage:  python3 gap.py [m_max]              (default 2)
"""
import sys
from fractions import Fraction as Fr

from kas6 import TT
from kasV import TYPES, Rmat, ulabel
import kasV
import gls                                    # registers k16, brings scan()
from gls import scan
from glscheck import check
import dagger
from symtype import sym_type


# ------------------------------------------------- the fibres S^m_0 as types

def register(mmax):
    out = []
    for m in range(2, mmax + 1):
        V = sym_type(m)
        TYPES[V.name] = V
        out.append(V.name)
    return out


def cV(V):
    """the scalar by which K_t acts on the fibre (gls.ct, inlined)."""
    M = Rmat(V, TT, 0)
    cs = {M[b][b].get((0, 0), 0) for b in range(V.dim)}
    assert len(cs) == 1
    return Fr(next(iter(cs)))


def gls_weight(V, lam):
    """the weight of [GLS] of the space of tensor fields T(V, lam)."""
    return (-(lam + cV(V)),) + tuple(max(V.wts))


def coeff_table(ts, vec):
    """the nullspace vector of an order-1 ansatz as {(i, a, b): c} for
    glscheck.check."""
    tab = {}
    for t, x in zip(ts, vec):
        if not x:
            continue
        (k, A), a, b = t
        assert k == 0 and len(A) == 1, f"not of order 1: {ulabel(t)}"
        tab[(A[0], a, b)] = Fr(x)
    return tab


def op_from(V2, tab):
    from superN import add, smul
    from kas6 import ALG
    D = [ALG.Dword((i,), 0) for i in range(6)]

    def P(phi):
        out = [{} for _ in range(V2.dim)]
        for (i, a, b), c in tab.items():
            if phi[a]:
                out[b] = add(out[b], smul(c, D[i](phi[a])))
        return out
    return P


def study(n1, n2, N=1, rng=8):
    V1, V2 = TYPES[n1], TYPES[n2]
    print(f"\n=== {n1} (dim {V1.dim}) --> {n2} (dim {V2.dim}), order {N} ===",
          flush=True)
    hits = scan('kas', n1, n2, N, rng)
    for lam1, lam2, ts, ns in hits:
        w1, w2 = gls_weight(V1, lam1), gls_weight(V2, lam2)
        where = dagger.in_list(6, (w1, w2))
        par = None
        if where:
            for name, npar, f in dagger.LISTS[6]:
                if name == where:
                    par = dagger.match(npar, f, (w1, w2))
        print(f"  lam = ({lam1}, {lam2}), dim of the space of operators = "
              f"{len(ns)}", flush=True)
        print(f"      in the weights of [GLS]: {w1} -> {w2}", flush=True)
        verdict = (f"family {where} at {dagger.fmt(par)}" if where
                   else "*** IN NO ENTRY OF THE LIST ***")
        print(f"      [GLS, Sec.10]: {verdict}", flush=True)
        if N != 1:
            continue
        for v in ns:
            print("      operator: ",
                  "  ".join(f"{ulabel(t)}:{x}"
                            for t, x in zip(ts, v) if x != 0), flush=True)
            tab = coeff_table(ts, v)
            P = op_from(V2, tab)
            check("   independent check ", 'kas', V1, V2, lam1, lam2, P, 1)
            check("   k(1|6)?           ", 'k16', V1, V2, lam1, lam2, P, 1)
            for d in (Fr(1, 2), Fr(-1, 2), Fr(1)):
                check(f"   control lam+{d}   ", 'kas', V1, V2,
                      lam1 + d, lam2 + d, P, 1)
    if not hits:
        print("  (no invariant operator at any weight of the scan)",
              flush=True)
    return hits


if __name__ == '__main__':
    mmax = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    names = ['F', 'L1'] + register(mmax)
    print("the fibres:", flush=True)
    for nm in names:
        V = TYPES[nm]
        print(f"  {nm}: dim {V.dim}, highest weight {tuple(max(V.wts))}, "
              f"c_V = {cV(V)}", flush=True)
    for m in range(len(names) - 1):
        study(names[m], names[m + 1])          # 1a): the gradients
        study(names[m + 1], names[m])          # 1a^dagger): the divergences
