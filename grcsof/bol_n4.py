#!/usr/bin/env python3
"""n = 4: the unary (Bol) operators invariant under the family h(a) = osp_a(4|2),
and their comparison with the binary GRC brackets.

h(a) is the family of k14family.py: h_1(a) = < t y_i + a sigma(y_i) >.
Everything is exact (Fractions), the weight lambda runs over a grid or is
symbolic; every solution found is re-verified against ALL the generators of
h(a) -- h_{-2}, h_{-1}, h_0, h_1(a), h_2(a) -- by an independent routine.
"""
import sys
from fractions import Fraction as Fr
from itertools import combinations
from superN import add, smul, mul, mono, parity, par_mon
from k14family import ALG, TY, CUB, QUAD, TOP, msk
from knclass import UnaryClassifier, nullspace

SIG = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]
NM = ALG.names


def wgens(a):
    """h_1(a) = < t y_i + a sigma(y_i) >."""
    return [add(TY[i], *[smul(a * SIG[i][j], CUB[j]) for j in range(4)])
            for i in range(4)]


def hfields(a):
    """ALL generating functions of h(a), by name."""
    out = {'1': mono(0, 0), 't': mono(1, 0)}
    for i in range(4):
        out[NM[i]] = mono(0, 1 << i)
    for (i, j) in combinations(range(4), 2):
        out[NM[i] + NM[j]] = mono(0, msk(i, j))
    for i, w in enumerate(wgens(a)):
        out['w_' + NM[i]] = w
    out['t2'] = add(mono(2, 0), smul(-2 * a, TOP))
    return out


def ulabel(t):
    k, A = t
    s = "".join("D" + NM[i] for i in A)
    if k:
        s = (f"dt^{k}" if k > 1 else "dt") + s
    return s or "1"


def uverify(coeffs, ts, lam, N, fields, tmax=4):
    """independent check that sum c_i P_i : F_lam --> F_{lam+N} is invariant."""
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


# the "top" Clifford monomials
def Top(n_pairs):
    """T_4 = D_xi1 D_eta1 D_xi2 D_eta2 - dt(D_xi1D_eta1 + D_xi2D_eta2) + dt^2,
    written as a coefficient vector in the order-4 unary ansatz."""
    ts = UnaryClassifier(ALG, [], 0).terms(4)
    c = [0] * len(ts)
    for i, (k, A) in enumerate(ts):
        if A == (0, 1, 2, 3):
            c[i] = 1
        elif A in ((0, 1), (2, 3)) and k == 1:
            c[i] = -1
        elif A == () and k == 2:
            c[i] = 1
    return ts, c


def scan(avals, Ns, grid, tmax=3):
    for a in avals:
        F = hfields(a)
        gens = list(F.values())
        print(f"=== a = {a}")
        for N in Ns:
            hits = []
            for lam in grid:
                U = UnaryClassifier(ALG, gens, lam)
                ts, ns = U.solve(N, tmax=tmax)
                if ns:
                    hits.append((lam, len(ns), ts, ns))
            if not hits:
                print(f"  order {N}: nothing on the grid")
            for lam, d, ts, ns in hits:
                print(f"  order {N}, lam = {lam}: dim = {d}")
                for v in ns:
                    print("      " + "  ".join(
                        f"{ulabel(t)}:{c}" for t, c in zip(ts, v) if c != 0))
                    bad = uverify(v, ts, lam, N, F)
                    print("      verified against all generators of h(a)"
                          if not bad else f"      FAILS on {bad}")
            sys.stdout.flush()


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    if what == 'scan':
        avals = [Fr(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 \
            else [Fr(0), Fr(2), Fr(1)]
        Ns = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 \
            else [1, 2, 3, 4, 5, 6]
        grid = [Fr(x, 2) for x in range(-16, 9)]
        scan(avals, Ns, grid)
    elif what == 'T4':
        # is T_4 invariant for every a?  symbolic a
        from sympy import Symbol, expand
        a = Symbol('a')
        ts, c = Top(4)
        print("T_4 =", "  ".join(f"{ulabel(t)}:{x}" for t, x in zip(ts, c) if x))
        for lam in [Fr(-3)]:
            bad = uverify(c, ts, lam, 4, hfields(a))
            print(f"  lam = {lam}: ",
                  "invariant for EVERY a" if not bad else f"fails on {bad}")
        # and at which lambda, for symbolic a, is the order-4 ansatz nonempty?
        from sympy import symbols
        l = Symbol('lam')
        U = UnaryClassifier(ALG, list(hfields(a).values()), l)
        ts, ns = U.solve(4, tmax=3)
        print("  generic lam, symbolic a: dim =", len(ns))
