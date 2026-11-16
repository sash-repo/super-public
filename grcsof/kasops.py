#!/usr/bin/env python3
"""n = 6: invariant differential operators between spaces of weighted densities
over k(1|6), invariant under

    as   = <K_1, Lambda^1, Lambda^2, Lambda^3_xi>          (16|16-dimensional),
    cas  = as + <K_t>,
    kas  = the partial Cartan prolong (g_- + g_0, Lambda^3_xi + t Lambda^1)_*,
    osp  = i(osp(6|2)) = g_- + g_0 + t Lambda^1 + <K_{t^2}>,

see kas6.py.  Ansatz (legitimate by Theorem "Peetre" of the paper, which needs
only h_- = g_-, true for all four): constant coefficients in the operators
d_t and D_i.

Two economies, both verified rather than assumed:

 (1) the Cartan subalgebra <K_{xi_i eta_i}> lies in all four algebras, so an
     invariant operator is of torus weight 0; we solve inside the weight-0 part
     of the ansatz and check against the unfiltered answer whenever that is
     affordable (--full);

 (2) equations are imposed for a Lie-generating subset only, and every solution
     found is then re-verified, by an independent routine, against the FULL
     list of generating functions of the algebra.
"""
import sys
from fractions import Fraction as Fr
from itertools import combinations

from superN import add, smul, mul, mono, parity, par_mon
from knclass import Classifier, UnaryClassifier, nullspace, f_id
from kas6 import (ALG, NAMES, full_gens, zero_weight, gens_as, gens_cas,
                  gens_kas, gens_osp)

NM = NAMES


# ------------------------------------------------------------- term lists

def unary_terms(N, filt=True):
    subs = [()] + [c for r in range(1, 7) for c in combinations(range(6), r)]
    ts = [((N - len(A)) // 2, A) for A in subs
          if N - len(A) >= 0 and (N - len(A)) % 2 == 0]
    return [t for t in ts if (not filt) or zero_weight(t[1])]


def binary_terms(N, filt=True):
    subs = [()] + [c for r in range(1, 7) for c in combinations(range(6), r)]
    out = []
    for A in subs:
        for Bs in subs:
            rest = N - len(A) - len(Bs)
            if rest < 0 or rest % 2:
                continue
            if filt and not zero_weight(A, Bs):
                continue
            for kA in range(rest // 2 + 1):
                out.append((kA, A, rest // 2 - kA, Bs))
    return out


def ulabel(t):
    k, A = t
    s = "".join("D" + NM[i] for i in A)
    if k:
        s = (f"dt^{k}" if k > 1 else "dt") + s
    return s or "1"


def blabel(t):
    kA, A, kB, Bs = t
    return f"{ulabel((kA, A))}(x){ulabel((kB, Bs))}"


# --------------------------------------------------------------- solvers

class U(UnaryClassifier):
    def __init__(self, gens, lam, filt=True):
        UnaryClassifier.__init__(self, ALG, gens, lam)
        self.filt = filt

    def terms(self, N):
        return unary_terms(N, self.filt)


class Bn(Classifier):
    def __init__(self, gens, lam1, lam2, filt=True, ymax=None):
        Classifier.__init__(self, ALG, gens, lam1, lam2)
        self.filt = filt
        self.ymax = ymax

    def build(self, N, tmax=None, zero=None):
        alg = ALG
        ts = binary_terms(N, self.filt)
        pB = N % 2
        lam = self.lam1 + self.lam2 + N
        if tmax is None:
            tmax = 2
        MS = [m for m in alg.monos(tmax)
              if self.ymax is None or bin(m[1]).count('1') <= self.ymax]
        Ps = [alg.Dword(A, kA) for (kA, A, kB, Bs) in ts]
        Qs = [alg.Dword(Bs, kB) for (kA, A, kB, Bs) in ts]
        pQ = [len(Bs) % 2 for (kA, A, kB, Bs) in ts]

        rows = {}
        for f in self.gens:
            pf = parity(f)
            sB = -1 if (pf and pB) else 1
            L1, L2, Lt = alg.L(f, self.lam1), alg.L(f, self.lam2), alg.L(f, lam)
            L1u = {m: L1(mono(*m)) for m in MS}
            L2v = {m: L2(mono(*m)) for m in MS}
            Pu, Qv, PLu, QLv = ({} for _ in range(4))
            for i in range(len(ts)):
                Pu[i] = {m: Ps[i](mono(*m)) for m in MS}
                Qv[i] = {m: Qs[i](mono(*m)) for m in MS}
                PLu[i] = {m: Ps[i](L1u[m]) for m in MS}
                QLv[i] = {m: Qs[i](L2v[m]) for m in MS}
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
                        res = add(Lt(Buv), smul(-sB, add(BLuv, smul(s2, BuLv))))
                        for k, val in res.items():
                            key = (f_id(f), m1, m2, k)
                            rows.setdefault(key, {})[i] = \
                                rows.setdefault(key, {}).get(i, 0) + val
        return ts, rows


# --------------------------------------------------------- verification

def uverify(coeffs, ts, lam, N, fields, tmax=3):
    """independent check that sum c_i P_i : F_lam --> F_{lam+N} is invariant
    under every function in the dict fields."""
    Ps = [ALG.Dword(A, k) for (k, A) in ts]
    pP = N % 2

    def P(u):
        out = {}
        for c, Pi in zip(coeffs, Ps):
            if c:
                out = add(out, smul(c, Pi(u)))
        return out

    bad = []
    for nm, f in fields.items():
        pf = parity(f)
        s = -1 if (pf and pP) else 1
        Ls, Lt = ALG.L(f, lam), ALG.L(f, lam + N)
        for m in ALG.monos(tmax):
            u = mono(*m)
            if add(Lt(P(u)), smul(-s, P(Ls(u)))):
                bad.append(nm)
                break
    return bad


def bverify(coeffs, ts, lam1, lam2, N, fields, tmax=2):
    Ps = [ALG.Dword(A, kA) for (kA, A, kB, Bs) in ts]
    Qs = [ALG.Dword(Bs, kB) for (kA, A, kB, Bs) in ts]
    pQ = [len(Bs) % 2 for (kA, A, kB, Bs) in ts]
    pB = N % 2
    lam = lam1 + lam2 + N
    MS = ALG.monos(tmax)

    def Bop(u, v, pu):
        out = {}
        for c, P, Q, pq in zip(coeffs, Ps, Qs, pQ):
            if c:
                sgn = -1 if (pq and pu) else 1
                out = add(out, smul(sgn * c, mul(P(u), Q(v))))
        return out

    bad = []
    for nm, f in fields.items():
        pf = parity(f)
        sB = -1 if (pf and pB) else 1
        L1, L2, Lt = ALG.L(f, lam1), ALG.L(f, lam2), ALG.L(f, lam)
        ok = True
        for m1 in MS:
            pu = par_mon(m1)
            s2 = -1 if (pf and pu) else 1
            for m2 in MS:
                u, v = mono(*m1), mono(*m2)
                lhs = Lt(Bop(u, v, pu))
                rhs = add(Bop(L1(u), v, (pu + pf) % 2), smul(s2, Bop(u, L2(v), pu)))
                if add(lhs, smul(-sB, rhs)):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            bad.append(nm)
    return bad


# -------------------------------------------------------------- reporting

def show_unary(v, ts):
    return "  ".join(f"{ulabel(t)}:{c}" for t, c in zip(ts, v) if c != 0)


def show_binary(v, ts):
    return "  ".join(f"{blabel(t)}:{c}" for t, c in zip(ts, v) if c != 0)


ALGS = {'as': gens_as, 'cas': gens_cas, 'kas': gens_kas, 'osp': gens_osp}

# Lie-generating subsets: g_- is automatic for the ansatz, Lambda^2 = o(6) and
# one highest-weight cubic generate Lambda^3_xi as an o(6)-module, so o(6) +
# one cubic + (for cas/kas/osp) the t-dependent generators suffice.
def eqn_gens(which):
    F = ALGS[which]()
    keys = ['1'] + NM + [NM[i] + NM[j] for i, j in combinations(range(6), 2)]
    if which in ('as', 'cas', 'kas'):
        keys += ['c0']
    if which in ('cas', 'kas', 'osp'):
        keys += ['t']
    if which in ('kas', 'osp'):
        keys += ['t' + NM[i] for i in range(6)]
    if which == 'osp':
        keys += ['t2']
    return [F[k] for k in keys]


if __name__ == '__main__':
    job = sys.argv[1]

    if job == 'unary':
        which = sys.argv[2]
        Ns = [int(x) for x in sys.argv[3].split(',')]
        lams = [Fr(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 \
            else [Fr(0), Fr(1), Fr(-1), Fr(-3), Fr(3, 7)]
        filt = '--full' not in sys.argv
        F = ALGS[which]()
        gens = eqn_gens(which)
        for N in Ns:
            for lam in lams:
                ts, ns = U(gens, lam, filt).solve(N, tmax=3)
                print(f"{which}: order {N}, lam = {lam}: "
                      f"{len(ts)} terms, dim = {len(ns)}")
                for v in ns:
                    print("     ", show_unary(v, ts))
                    bad = uverify(v, ts, lam, N, F)
                    print("      verified against all generators"
                          if not bad else f"      FAILS on {bad}")
                sys.stdout.flush()

    elif job == 'binary':
        which = sys.argv[2]
        Ns = [int(x) for x in sys.argv[3].split(',')]
        l1 = Fr(sys.argv[4]) if len(sys.argv) > 4 else Fr(3, 7)
        l2 = Fr(sys.argv[5]) if len(sys.argv) > 5 else Fr(-5, 11)
        filt = '--full' not in sys.argv
        ymax = 3 if '--ymax3' in sys.argv else None
        F = ALGS[which]()
        gens = eqn_gens(which)
        for N in Ns:
            ts, ns = Bn(gens, l1, l2, filt, ymax).solve(N, tmax=2)
            print(f"{which}: order {N}, lam = ({l1},{l2}): "
                  f"{len(ts)} terms, dim = {len(ns)}")
            for v in ns:
                print("     ", show_binary(v, ts))
                bad = bverify(v, ts, l1, l2, N, F)
                print("      verified against all generators"
                      if not bad else f"      FAILS on {bad}")
            sys.stdout.flush()
