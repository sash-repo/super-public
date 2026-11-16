#!/usr/bin/env python3
"""k(1|5): the binary i(osp(5|2))-invariant operators, with the torus economy.

The maximal toral subalgebra <K_{xi_i eta_i} | i = 1,2> lies in i(osp(5|2)) and
acts on D_xi_i, D_eta_i, D_th, d_t with the weights +1, -1, 0, 0 (Lemma "no
invariant operators of odd order" of the paper).  Hence an invariant operator
lies in the weight-0 part of the ansatz, and we may solve there; the answer is
cross-checked against the unfiltered run (n5.py) in the orders where the latter
is affordable, and every solution found is fed to the independent invariance
test verify_kn.check against ALL 23 generating functions of i(osp(5|2)).

Note that D_th has weight 0, so, unlike the case of even n, terms of odd order
survive the filter: for odd n odd-order operators are not excluded a priori.

Usage:  n5bin.py 1,2,3,4,5,6 [sym] [tmax]
"""
import sys
from fractions import Fraction as Fr
from itertools import combinations

from superN import add, smul, mul, mono, parity, par_mon
from knclass import Classifier, f_id, nullspace, label
from n5 import ALG
from verify_kn import check, osp_fields

NM = ALG.names
PAIRS = [(1, 2), (3, 4)]          # (xi_i, eta_i)


def wt(*subsets):
    w = [0, 0]
    for A in subsets:
        for k, (x, e) in enumerate(PAIRS):
            w[k] += (x in A) - (e in A)
    return w


def zero_weight(*subsets):
    return all(x == 0 for x in wt(*subsets))


def binary_terms(N, filt=True):
    subs = [()] + [c for r in range(1, 6) for c in combinations(range(5), r)]
    out = []
    for A in subs:
        for B in subs:
            rest = N - len(A) - len(B)
            if rest < 0 or rest % 2:
                continue
            if filt and not zero_weight(A, B):
                continue
            for kA in range(rest // 2 + 1):
                out.append((kA, A, rest // 2 - kA, B))
    return out


class Bn(Classifier):
    def __init__(self, gens, lam1, lam2, filt=True):
        Classifier.__init__(self, ALG, gens, lam1, lam2)
        self.filt = filt

    def build(self, N, tmax=None, zero=None):
        alg = ALG
        ts = binary_terms(N, self.filt)
        pB = N % 2
        lam = self.lam1 + self.lam2 + N
        tmax = 2 if tmax is None else tmax
        MS = alg.monos(tmax)
        Ps = [alg.Dword(A, kA) for (kA, A, kB, B) in ts]
        Qs = [alg.Dword(B, kB) for (kA, A, kB, B) in ts]
        pQ = [len(B) % 2 for (kA, A, kB, B) in ts]

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


def gens_osp():
    return [mono(1, 1 << i) for i in range(5)]


def run(Ns, l1, l2, tmax=2, filt=True, verify=True, vtmax=3):
    fields = osp_fields(ALG)
    for N in Ns:
        C = Bn(gens_osp(), l1, l2, filt=filt)
        ts, ns = C.solve(N, tmax=tmax)
        print(f"order {N}: {len(ts)} terms (weight-0 filter={filt}), "
              f"dim = {len(ns)}")
        sys.stdout.flush()
        for v in ns:
            print("  solution:")
            for t, c in zip(ts, v):
                if c != 0:
                    print(f"     {label(t, NM)} : {c}")
            sys.stdout.flush()
            if verify:
                check(ALG, ts, v, l1, l2, N, fields, tmax=vtmax)
                sys.stdout.flush()


if __name__ == '__main__':
    Ns = [int(x) for x in sys.argv[1].split(',')]
    sym = len(sys.argv) > 2 and sys.argv[2] == 'sym'
    if sym:
        from sympy import symbols
        l1, l2 = symbols('l1 l2')
    else:
        l1, l2 = Fr(3, 7), Fr(-5, 11)
    tmax = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    print(f"k(1|5), i(osp(5|2)), weights {l1}, {l2}, tmax={tmax}")
    run(Ns, l1, l2, tmax=tmax, verify=not sym)
