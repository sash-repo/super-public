#!/usr/bin/env python3
"""k(1|5): the unary i(osp(5|2))-invariant operators F_lam --> F_{lam+N}
(the analogs of the Bol operators), and the check of the predicted answer

    T_5 o d_t^k : F_{-4-k} --> F_{1+k},    N = 5 + 2k,
    T_5 := D_th (D_xi1 D_eta1 - d_t)(D_xi2 D_eta2 - d_t),

which is [BLS1, Thm 1.4.2] read at n = 5 (lam = 1-k-n, mu = k+1).

Usage:  n5unary.py scan  N1,N2,...       -- dimension over a grid of weights
        n5unary.py t5    kmax            -- build T_5 o d_t^k and test it
"""
import sys
from fractions import Fraction as Fr
from itertools import combinations

from superN import add, smul, mono, parity, compose, dtn
from knclass import UnaryClassifier, osp_gens, label
from n5 import ALG
from verify_kn import osp_fields

NM = ALG.names
GRID = [Fr(x, 2) for x in range(-16, 9)]


# --------------------------------------------------------------- verification

def check_unary(P, lam, N, fields, tmax=4, verbose=True):
    """L^{lam+N}_f(P(u)) = (-1)^{p(f)p(P)} P(L^{lam}_f(u)) for all f, all u."""
    pP = N % 2
    bad = []
    for nm, f in fields.items():
        pf = parity(f)
        s = -1 if (pf and pP) else 1
        Ls, Lt = ALG.L(f, lam), ALG.L(f, lam + N)
        for m in ALG.monos(tmax):
            u = mono(*m)
            if add(Lt(P(u)), smul(-s, P(Ls(u)))):
                bad.append((nm, m))
                break
    if verbose:
        print("   NOT invariant under K_%s" % bad[0][0] if bad else
              "   invariant under all %d generators of i(osp(5|2)) "
              "(monomials up to t^%d)" % (len(fields), tmax))
    return not bad


# ------------------------------------------------------------------- T_5

def T5_op(k):
    """T_5 o d_t^k as a map on superfunctions."""
    Dth = ALG.Dword((0,), 0)
    dt = dtn(1)
    P1 = lambda u: add(ALG.Dword((1, 2), 0)(u), smul(-1, dt(u)))
    P2 = lambda u: add(ALG.Dword((3, 4), 0)(u), smul(-1, dt(u)))
    dk = dtn(k)
    return lambda u: Dth(P1(P2(dk(u))))


def main_scan(Ns):
    fields = osp_fields(ALG)
    for N in Ns:
        print(f"--- order {N}")
        for lam in GRID:
            U = UnaryClassifier(ALG, osp_gens(ALG), lam)
            ts, ns = U.solve(N, tmax=2)
            if ns:
                print(f"  lam = {lam}: dim = {len(ns)}")
                for v in ns:
                    print("     ", {label((k, A), NM) if False else
                                    ("dt^%d" % k if not A else
                                     ("dt^%d " % k if k else "") +
                                     "".join("D" + NM[i] for i in A)): str(c)
                                    for (k, A), c in zip(ts, v) if c != 0})
                sys.stdout.flush()
        sys.stdout.flush()


def main_t5(kmax):
    fields = osp_fields(ALG)
    for k in range(kmax + 1):
        N, lam = 5 + 2 * k, Fr(-4 - k)
        P = T5_op(k)
        print(f"T_5 o d_t^{k}: order {N}, F_{lam} --> F_{lam + N}")
        ok = check_unary(P, lam, N, fields)
        # and that it is invariant at NO other weight
        others = [l for l in GRID if l != lam
                  and check_unary(P, l, N, fields, tmax=2, verbose=False)]
        print("   other weights on the grid at which it is invariant:",
              others or "none")
        sys.stdout.flush()


if __name__ == '__main__':
    if sys.argv[1] == 'scan':
        main_scan([int(x) for x in sys.argv[2].split(',')])
    else:
        main_t5(int(sys.argv[2]))
