#!/usr/bin/env python3
"""Comparison of Section 6 of the paper with the kas-invariant unary operators of [GLS]
(Grozman-Leites-Shchepochkina, "Invariant operators on supermanifolds and
standard models", arXiv:math/0202193), Section 10.

[GLS] describe the SINGULAR VECTORS in I(V) = U(g) (x)_{U(g_0)} V for
g = kas and g = k(1|6) in the standard grading; since I(V)^* = T(V^*), a
singular vector of weight mu in I(V^lam) is the same thing as an invariant
differential operator T(V_lam^*) --> T(V_mu^*), of order = the degree of the
singular vector.

Here we compute, with the engine of kasV.py, the kas- and k(1|6)-invariant
unary operators between the tensor fields T(V, lam) of Subsection 6.6 -- now with
the two weights lam1, lam2 INDEPENDENT, since for kas (unlike for as) the
weight is not invisible.  For each pair (V1, V2) and each order N we

  (1) determine the line lam2 = lam1 + c on which K_t-invariance is possible
      (K_t alone, cheap), and
  (2) solve the full invariance problem along that line.

Usage:  python3 gls.py ct                      the K_t-scalars c_V
        python3 gls.py scan <alg> <V1> <V2> <N> [lam-range]
        python3 gls.py all <alg>               all the cases of Subsection 6.6
"""
import sys
from fractions import Fraction as Fr

from kas6 import ALG, NAMES, gens_kas, CUB_ETA, named, TT
from kasbin import Echelon, testmons, ROOTS
import kasV
from kasV import TYPES, uterms, ubuild, usolve, ulabel, Rmat


# ------------------------------------------------- k(1|6) = kas + Lambda^3_eta

def gens_k16():
    """all of k(1|6) in degrees <= 1: kas + Lambda^3_eta."""
    out = gens_kas()
    out.update(named(CUB_ETA, 'e'))
    return out


kasV.ALGS['k16'] = gens_k16
from kasbin import START                                   # noqa: E402
START['k16'] = ROOTS + ['c0', 'e0', 't', 'txi1', 'teta1']


# ------------------------------------------------------------- the K_t scalar

def ct(V):
    """K_t acts on T(V, 0) by K_t + c_V, with c_V a scalar; return c_V."""
    M = Rmat(V, TT, 0)
    c = None
    for b in range(V.dim):
        for a in range(V.dim):
            e = M[b][a]
            val = e.get((0, 0), 0) if e else 0
            if a == b:
                if c is None:
                    c = Fr(val)
                elif c != val:
                    raise AssertionError(f"{V.name}: K_t not scalar")
            elif e:
                raise AssertionError(f"{V.name}: K_t not diagonal")
    return c


def kt_line(V1, V2, N):
    """the constant c with lam2 = lam1 + c forced by K_t-invariance, or None
    if K_t kills the ansatz for every c."""
    ts = uterms(V1, V2, N)
    if not ts:
        return None, 0
    F = gens_k16()
    for c in [Fr(n, 2) for n in range(-40, 41)]:
        ns = ubuild([F['t']], V1, V2, N, Fr(0), c, testmons(2, 4), ts)
        if ns:
            return c, len(ts)
    return None, len(ts)


CASES = [('F', 'L1'), ('L1', 'F'), ('L1', 'L1'),
         ('Sp', 'Sm'), ('Sm', 'Sp'), ('Sp', 'Sp'), ('Sm', 'Sm'),
         ('F', 'L3p'), ('L3p', 'F'), ('F', 'L3m'), ('L3m', 'F'),
         ('L3p', 'L3p'), ('L3m', 'L3m'), ('L3p', 'L3m'), ('L3m', 'L3p'),
         ('F', 'F')]


def scan(alg, n1, n2, N, rng=8, quiet=False):
    V1, V2 = TYPES[n1], TYPES[n2]
    c, nterms = kt_line(V1, V2, N)
    if c is None:
        print(f"{alg}: {n1} -> {n2}, order {N}: "
              f"{nterms} terms, no K_t-line  => dim 0 for every weight",
              flush=True)
        return []
    hits = []
    for num in range(-2 * rng, 2 * rng + 1):
        lam1 = Fr(num, 2)
        ts, ns = usolve(alg, V1, V2, N, lam1, lam1 + c, quiet=True)
        if ns:
            hits.append((lam1, lam1 + c, ts, ns))
    print(f"{alg}: {n1} -> {n2}, order {N}: {nterms} terms, "
          f"K_t-line lam2 = lam1 + {c}; "
          f"{len(hits)} weight(s) with a nonzero answer "
          f"(scan lam1 in [-{rng}, {rng}], step 1/2)", flush=True)
    for lam1, lam2, ts, ns in hits:
        print(f"    lam1 = {lam1}, lam2 = {lam2}: dim = {len(ns)}", flush=True)
        for v in ns:
            print("       ", "  ".join(f"{ulabel(t)}:{x}"
                                       for t, x in zip(ts, v) if x != 0),
                  flush=True)
    return hits


if __name__ == '__main__':
    job = sys.argv[1]
    if job == 'ct':
        for n, V in sorted(TYPES.items()):
            print(f"  c_{n} = {ct(V)}   (dim {V.dim})")
    elif job == 'scan':
        alg, n1, n2 = sys.argv[2], sys.argv[3], sys.argv[4]
        for N in [int(x) for x in sys.argv[5].split(',')]:
            rng = int(sys.argv[6]) if len(sys.argv) > 6 else 8
            scan(alg, n1, n2, N, rng)
    elif job == 'all':
        alg = sys.argv[2]
        Ns = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 \
            else [1, 2, 3]
        for n1, n2 in CASES:
            for N in Ns:
                scan(alg, n1, n2, N, 8)
