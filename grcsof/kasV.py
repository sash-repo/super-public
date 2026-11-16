#!/usr/bin/env python3
"""n = 6: invariant differential operators between spaces of TENSOR FIELDS
of type V over kas, for V a kas_0-module of dimension > 1.

    T(V, lam) = V-valued superfunctions,
    (L_f phi)_b = K_f(phi_b) + lam d_t(f) phi_b + sum_a rho(a(f)^T)_{ba} phi_a

(see kastypes.py).  Unary operators

    P : T(V_1,lam_1) --> T(V_2,lam_2),   (P phi)_b = sum_{a,w} c_{w,a,b} w(phi_a),

and binary ones

    B : T(V_1,lam_1) (x) T(V_2,lam_2) --> T(V_3,lam_3),

with w running over the words d_t^k D_{i_1}...D_{i_r} of a given order N (the
constant-coefficient ansatz, legitimate by the Peetre-type theorem of the paper
since h_- = g_- for all the subalgebras h considered).

The two economies of kasbin.py are kept: the ansatz is cut to the part of
torus weight 0, the equations are imposed for a small Lie-generating subset,
and EVERY solution is re-verified against the FULL list of generators.
"""
import sys
from fractions import Fraction as Fr
from itertools import combinations

from superN import add, smul, mul, mono, parity, par_mon, d_t
from kas6 import (ALG, NAMES, wt_of_index, gens_as, gens_cas, gens_kas,
                  gens_osp, gens_asop, CUB_XI, CUB_ETA)
from kastens import amat, transpose
from kastypes import Type, lam_type, submodule, restrict, spin_type, rho_of
from kasbin import Echelon, testmons, START

NM = NAMES
ALGS = {'as': gens_as, 'cas': gens_cas, 'kas': gens_kas, 'osp': gens_osp,
        'asop': gens_asop}

# ------------------------------------------------------------- the types

TRIV = Type('F', 1, [[[Fr(0)]] for _ in range(16)], ['1'])
L1 = lam_type(1)
L2 = lam_type(2)
L3 = lam_type(3)


def _split_L3():
    """the two 10-dimensional halves of Lambda^3; the one containing the
    coordinate vector of weight (1,1,1) (i.e. matching Lambda^3_xi subset as)
    is called L3p."""
    seedp = [Fr(1) if w == (1, 1, 1) else Fr(0) for w in L3.wts]
    seedm = [Fr(1) if w == (-1, -1, -1) else Fr(0) for w in L3.wts]
    return (restrict(L3, submodule(L3, seedp), 'L3p'),
            restrict(L3, submodule(L3, seedm), 'L3m'))


L3p, L3m = _split_L3()
SP = spin_type(True)
SM = spin_type(False)

TYPES = {'F': TRIV, 'L1': L1, 'L2': L2, 'L3': L3, 'L3p': L3p, 'L3m': L3m,
         'Sp': SP, 'Sm': SM}


# --------------------------------------------------------- the R-matrices

_R = {}


def Rmat(V, f, lam):
    key = (V.name, tuple(sorted(f.items(), key=str)), lam)
    r = _R.get(key)
    if r is None:
        M = rho_of(V, transpose(amat(f))) if V.dim > 1 or True else None
        if lam:
            dtf = smul(lam, d_t(f))
            if dtf:
                M = [row[:] for row in M]
                for b in range(V.dim):
                    M[b][b] = add(M[b][b], dtf)
        _R[key] = r = M
    return r


_K = {}


def Kop(f):
    key = tuple(sorted(f.items(), key=str))
    r = _K.get(key)
    if r is None:
        r = _K[key] = ALG.K(f)
    return r


def Lf(V, f, lam, phi):
    Kf = Kop(f)
    R = Rmat(V, f, lam)
    out = []
    for b in range(V.dim):
        s = Kf(phi[b])
        for a in range(V.dim):
            if R[b][a] and phi[a]:
                s = add(s, mul(R[b][a], phi[a]))
        out.append(s)
    return out


def check_module(V, lam, gens, tmax=1):
    """L_{{f,g}} = L_f L_g - (-1)^{p(f)p(g)} L_g L_f."""
    bad = []
    for nf, f in gens.items():
        for ng, g in gens.items():
            fg = ALG.kb(f, g)
            s = -1 if (parity(f) and parity(g)) else 1
            hit = False
            for b0 in range(V.dim):
                for m in ALG.monos(tmax):
                    phi = [{} for _ in range(V.dim)]
                    phi[b0] = mono(*m)
                    lhs = Lf(V, fg, lam, phi) if fg else [{}] * V.dim
                    r1 = Lf(V, f, lam, Lf(V, g, lam, phi))
                    r2 = Lf(V, g, lam, Lf(V, f, lam, phi))
                    if any(add(x, smul(-1, add(y, smul(-s, z))))
                           for x, y, z in zip(lhs, r1, r2)):
                        bad.append((nf, ng, b0, m))
                        hit = True
                        break
                if hit:
                    break
    return bad


# ------------------------------------------------------------- the ansatz

def words(N):
    subs = [()] + [c for r in range(1, 7) for c in combinations(range(6), r)]
    return [((N - len(A)) // 2, A) for A in subs
            if N - len(A) >= 0 and (N - len(A)) % 2 == 0]


def wD(A):
    w = [0, 0, 0]
    for i in A:
        for k, x in enumerate(wt_of_index(i)):
            w[k] += x
    return tuple(w)


def wlabel(k, A):
    s = "".join("D" + NM[i] for i in A)
    if k:
        s = (f"dt^{k}" if k > 1 else "dt") + s
    return s or "1"


def uterms(V1, V2, N):
    """(word, a, b) of total torus weight 0."""
    out = []
    for w in words(N):
        d = wD(w[1])
        for a in range(V1.dim):
            for b in range(V2.dim):
                if all(d[k] + V2.wts[b][k] - V1.wts[a][k] == 0 for k in range(3)):
                    out.append((w, a, b))
    return out


def bterms(V1, V2, V3, N):
    subs = [()] + [c for r in range(1, 7) for c in combinations(range(6), r)]
    out = []
    for A in subs:
        for Bs in subs:
            rest = N - len(A) - len(Bs)
            if rest < 0 or rest % 2:
                continue
            d = [x + y for x, y in zip(wD(A), wD(Bs))]
            for kA in range(rest // 2 + 1):
                for a1 in range(V1.dim):
                    for a2 in range(V2.dim):
                        for b in range(V3.dim):
                            if all(d[k] + V3.wts[b][k] - V1.wts[a1][k]
                                   - V2.wts[a2][k] == 0 for k in range(3)):
                                out.append(((kA, A, rest // 2 - kA, Bs),
                                            a1, a2, b))
    return out


def ulabel(t):
    (k, A), a, b = t
    return f"{wlabel(k,A)}[{a}->{b}]"


def blabel(t):
    (kA, A, kB, Bs), a1, a2, b = t
    return f"{wlabel(kA,A)}(x){wlabel(kB,Bs)}[{a1},{a2}->{b}]"


# ------------------------------------------------------------- the solver

def ubuild(gens, V1, V2, N, lam1, lam2, MS, ts, check=None, names=None):
    pP = N % 2
    Ws = {}
    for (k, A) in words(N):
        Ws[(k, A)] = ALG.Dword(A, k)
    ech = Echelon(len(ts))
    failed = []
    for gi, f in enumerate(gens):
        pf = parity(f)
        sP = -1 if (pf and pP) else 1
        R1, R2 = Rmat(V1, f, lam1), Rmat(V2, f, lam2)
        Kf = Kop(f)
        for a0 in range(V1.dim):
            for m in MS:
                u = mono(*m)
                # L_f (u e_{a0})
                Lsrc = [{} for _ in range(V1.dim)]
                Lsrc[a0] = Kf(u)
                for a in range(V1.dim):
                    if R1[a][a0]:
                        Lsrc[a] = add(Lsrc[a], mul(R1[a][a0], u))
                loc = {}          # (b, key) -> {i: coeff}
                for i, (w, a, b) in enumerate(ts):
                    if a != a0:
                        continue
                    Wu = Ws[w](u)
                    if not Wu:
                        continue
                    # + L^{V2}_f applied: K_f part, into slot b
                    for kk, vv in Kf(Wu).items():
                        loc.setdefault((b, kk), {})[i] = \
                            loc.setdefault((b, kk), {}).get(i, 0) + vv
                    # + R2 part: goes into slot c
                    for c in range(V2.dim):
                        if R2[c][b]:
                            for kk, vv in mul(R2[c][b], Wu).items():
                                loc.setdefault((c, kk), {})[i] = \
                                    loc.setdefault((c, kk), {}).get(i, 0) + vv
                for i, (w, a, b) in enumerate(ts):
                    if not Lsrc[a]:
                        continue
                    for kk, vv in Ws[w](Lsrc[a]).items():
                        loc.setdefault((b, kk), {})[i] = \
                            loc.setdefault((b, kk), {}).get(i, 0) - sP * vv
                for r in loc.values():
                    row = [r.get(i, 0) for i in range(len(ts))]
                    if not any(row):
                        continue
                    if check is None:
                        ech.add(row)
                    else:
                        for v in check:
                            if sum(x * y for x, y in zip(row, v) if x and y):
                                failed.append(names[gi] if names else str(gi))
                                return failed
    return failed if check is not None else ech.nullspace()


def usolve(which, V1, V2, N, lam1=0, lam2=0, tmax=1, ymax=3, vtmax=2,
           quiet=False):
    F = ALGS[which]()
    keys = list(START[which])
    ts = uterms(V1, V2, N)
    if not ts:
        return ts, []
    MS = testmons(tmax, ymax)
    for _ in range(6):
        gens = [F[k] for k in keys]
        ns = ubuild(gens, V1, V2, N, lam1, lam2, MS, ts)
        if not quiet:
            print(f"  [{which} {V1.name}->{V2.name} N={N}] {len(ts)} terms, "
                  f"{len(keys)} eq.gens -> dim = {len(ns)}", flush=True)
        vnames = sorted(F)
        bad = set(ubuild([F[k] for k in vnames], V1, V2, N, lam1, lam2,
                         testmons(vtmax, 6), ts, check=ns, names=vnames)) \
            if ns else set()
        if not bad:
            return ts, ns
        new = bad - set(keys)
        if new:
            keys += sorted(new)
        else:
            ymax += 1
            tmax += 1
            MS = testmons(tmax, ymax)
    raise RuntimeError("did not stabilize")


def bbuild(gens, V1, V2, V3, N, lams, MS, ts, check=None, names=None):
    """the invariance equations for the binary ansatz.

    All evaluations of the words of the ansatz on the test monomials are
    hoisted out of the inner loops; this is what makes the V-valued case
    affordable."""
    lam1, lam2, lam3 = lams
    pB = N % 2
    WD = {}
    for t in ts:
        kA, A, kB, Bs = t[0]
        WD.setdefault((kA, A), ALG.Dword(A, kA))
        WD.setdefault((kB, Bs), ALG.Dword(Bs, kB))
    WK = sorted(WD)
    WI = {w: i for i, w in enumerate(WK)}
    Wu = [[WD[w](mono(*m)) for m in MS] for w in WK]        # w(monomial)
    tidx = [(WI[(t[0][0], t[0][1])], WI[(t[0][2], t[0][3])],
             len(t[0][3]) % 2, t[1], t[2], t[3]) for t in ts]
    # terms sorted by their source index pair, to skip quickly
    bysrc = {}
    for i, (iP, iQ, pQ, b1, b2, b) in enumerate(tidx):
        bysrc.setdefault((b1, b2), []).append(i)
    ech = Echelon(len(ts))
    failed = []
    for gi, f in enumerate(gens):
        pf = parity(f)
        sB = -1 if (pf and pB) else 1
        R1, R2, R3 = Rmat(V1, f, lam1), Rmat(V2, f, lam2), Rmat(V3, f, lam3)
        Kf = Kop(f)
        # L_f (v e_{a2}) and every word applied to it
        LVw = {}
        for a2 in range(V2.dim):
            for i2, m2 in enumerate(MS):
                v = mono(*m2)
                Lv = [{} for _ in range(V2.dim)]
                Lv[a2] = Kf(v)
                for a in range(V2.dim):
                    if R2[a][a2]:
                        Lv[a] = add(Lv[a], mul(R2[a][a2], v))
                LVw[(a2, i2)] = [[WD[w](Lv[b]) if Lv[b] else {} for w in WK]
                                 for b in range(V2.dim)]
        for a1 in range(V1.dim):
            for i1, m1 in enumerate(MS):
                u = mono(*m1)
                pu = par_mon(m1)
                s2 = -1 if (pf and pu) else 1
                pLu = (pu + pf) % 2
                Lu = [{} for _ in range(V1.dim)]
                Lu[a1] = Kf(u)
                for a in range(V1.dim):
                    if R1[a][a1]:
                        Lu[a] = add(Lu[a], mul(R1[a][a1], u))
                LUw = [[WD[w](Lu[b]) if Lu[b] else {} for w in WK]
                       for b in range(V1.dim)]
                for a2 in range(V2.dim):
                    for i2 in range(len(MS)):
                        LW = LVw[(a2, i2)]
                        loc = {}
                        for i, (iP, iQ, pQ, b1, b2, b) in enumerate(tidx):
                            sgn = -1 if (pQ and pu) else 1
                            # LHS: L^{V3}_f ( B(u e_{a1}, v e_{a2}) )
                            if b1 == a1 and b2 == a2:
                                X = smul(sgn, mul(Wu[iP][i1], Wu[iQ][i2]))
                                if X:
                                    for kk, vv in Kf(X).items():
                                        r = loc.setdefault((b, kk), {})
                                        r[i] = r.get(i, 0) + vv
                                    for c in range(V3.dim):
                                        if R3[c][b]:
                                            for kk, vv in mul(R3[c][b], X).items():
                                                r = loc.setdefault((c, kk), {})
                                                r[i] = r.get(i, 0) + vv
                            # RHS: B(L_f u e_{a1}, v e_{a2})
                            if b2 == a2 and LUw[b1][iP]:
                                sgn1 = -1 if (pQ and pLu) else 1
                                for kk, vv in smul(sgn1, mul(LUw[b1][iP],
                                                             Wu[iQ][i2])).items():
                                    r = loc.setdefault((b, kk), {})
                                    r[i] = r.get(i, 0) - sB * vv
                            # RHS: B(u e_{a1}, L_f v e_{a2})
                            if b1 == a1 and LW[b2][iQ]:
                                for kk, vv in smul(sgn * s2,
                                                   mul(Wu[iP][i1],
                                                       LW[b2][iQ])).items():
                                    r = loc.setdefault((b, kk), {})
                                    r[i] = r.get(i, 0) - sB * vv
                        for r in loc.values():
                            row = [r.get(i, 0) for i in range(len(ts))]
                            if not any(row):
                                continue
                            if check is None:
                                ech.add(row)
                            else:
                                for vv2 in check:
                                    if sum(x * y for x, y in zip(row, vv2)
                                           if x and y):
                                        failed.append(names[gi] if names
                                                      else str(gi))
                                        return failed
    return failed if check is not None else ech.nullspace()


def bsolve(which, V1, V2, V3, N, lams=(0, 0, 0), tmax=1, ymax=2, vtmax=2,
           quiet=False):
    F = ALGS[which]()
    keys = list(START[which])
    ts = bterms(V1, V2, V3, N)
    if not ts:
        return ts, []
    MS = testmons(tmax, ymax)
    for _ in range(6):
        gens = [F[k] for k in keys]
        ns = bbuild(gens, V1, V2, V3, N, lams, MS, ts)
        if not quiet:
            print(f"  [{which} {V1.name}x{V2.name}->{V3.name} N={N}] "
                  f"{len(ts)} terms, {len(keys)} eq.gens -> dim = {len(ns)}",
                  flush=True)
        vnames = sorted(F)
        bad = set(bbuild([F[k] for k in vnames], V1, V2, V3, N, lams,
                         testmons(vtmax, 4), ts, check=ns, names=vnames)) \
            if ns else set()
        if not bad:
            return ts, ns
        new = bad - set(keys)
        if new:
            keys += sorted(new)
        else:
            ymax += 1
            tmax += 1
            MS = testmons(tmax, ymax)
    raise RuntimeError("did not stabilize")
