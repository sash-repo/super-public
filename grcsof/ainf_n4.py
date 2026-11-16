#!/usr/bin/env python3
"""Two supplements to bol_n4.py:

  (A) the degenerate member a = infinity, i.e. h_1 = < sigma(y_i) > = the cubics
      (this is osp_{-1}(4|2) in the notation of [BLS1], their Thm 1.6(ii));
  (B) the automorphism xi_1 <--> eta_1 of k(1|4) conjugates h(a) into h(-a),
      which is why the parameter of the *embedding* is a^2 (equivalently
      alpha + 1/alpha), not alpha.
"""
import sys
from fractions import Fraction as Fr
from itertools import combinations
from sympy import Symbol, expand, Matrix, S
from superN import add, smul, mono, clean
from k14family import ALG, TY, CUB, TOP, msk
from knclass import UnaryClassifier
from bol_n4 import wgens, ulabel, uverify, SIG, NM


def hinf():
    """generating functions of h(infinity): h_1 = <cubics>, h_2 = <y1y2y3y4>."""
    out = {'1': mono(0, 0), 't': mono(1, 0)}
    for i in range(4):
        out[NM[i]] = mono(0, 1 << i)
    for (i, j) in combinations(range(4), 2):
        out[NM[i] + NM[j]] = mono(0, msk(i, j))
    for i in range(4):
        out['s_' + NM[i]] = CUB[i]
    out['top'] = TOP
    return out


def swap(u):
    """the substitution xi_1 <--> eta_1 (bits 0 and 1) on superfunctions."""
    out = {}
    for (a, m), c in u.items():
        b0, b1 = m >> 0 & 1, m >> 1 & 1
        m2 = (m & ~3) | (b0 << 1) | b1
        # both bits present: y1y2 -> y2y1 = -y1y2
        s = -1 if (b0 and b1) else 1
        out[(a, m2)] = out.get((a, m2), 0) + s * c
    return clean(out)


if __name__ == '__main__':
    job = sys.argv[1] if len(sys.argv) > 1 else 'inf'

    if job == 'inf':
        F = hinf()
        gens = list(F.values())
        grid = [Fr(x, 2) for x in range(-16, 9)]
        for N in [1, 2, 3, 4, 5, 6]:
            hits = []
            for lam in grid:
                ts, ns = UnaryClassifier(ALG, gens, lam).solve(N, tmax=3)
                if ns:
                    hits.append((lam, ts, ns))
            if not hits:
                print(f"  order {N}: nothing on the grid")
            for lam, ts, ns in hits:
                print(f"  order {N}, lam = {lam}: dim = {len(ns)}")
                for v in ns:
                    print("      " + "  ".join(f"{ulabel(t)}:{c}"
                                               for t, c in zip(ts, v) if c != 0))
                    bad = uverify(v, ts, lam, N, F)
                    print("      verified" if not bad else f"      FAILS {bad}")
            sys.stdout.flush()
        # symbolic lambda: is d_t^k invariant for EVERY lambda?
        lam = Symbol('lam')
        for N in [2, 4]:
            ts, ns = UnaryClassifier(ALG, gens, lam).solve(N, tmax=3)
            print(f"  symbolic lam, order {N}: dim = {len(ns)}",
                  [{ulabel(t): c for t, c in zip(ts, v) if c != 0} for v in ns])

    elif job == 'swap':
        a = Symbol('a')
        W = wgens(a)
        Wm = wgens(-a)
        print("xi_1 <-> eta_1 applied to the generators of h_1(a):")
        span = [Wm[i] for i in range(4)]
        keys = sorted({k for u in span + [swap(w) for w in W] for k in u},
                      key=str)

        def vec(u):
            return [S(expand(u.get(k, 0))) for k in keys]
        M = Matrix([vec(u) for u in span])
        for i, w in enumerate(W):
            sw = swap(w)
            M2 = M.col_join(Matrix([vec(sw)]))
            print(f"   swap(w_{NM[i]}(a)) = {ALG.show(sw)}"
                  f"   in h_1(-a): {M.rank() == M2.rank()}")
        # h_2
        E = add(mono(2, 0), smul(-2 * a, TOP))
        Em = add(mono(2, 0), smul(2 * a, TOP))
        print("   swap(t^2-2a*top) =", ALG.show(swap(E)),
              "   equals t^2+2a*top:", not clean({k: expand(v) for k, v in
                                                  add(swap(E), smul(-1, Em)).items()}))
