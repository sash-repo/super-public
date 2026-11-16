#!/usr/bin/env python3
r"""eulerchk.py --- the two-term character identity
behind Theorems 12.1.2 and 12.1.4.  For the atypical families
  osp(3|2):   lambda_n = (n+1) eps + n delta      (m = 2n+2)
  osp_a(4|2): lambda_k = (k+1, k-1, k-1)           (p = q = r = k)
we check, as identities of Laurent polynomials, that Kac's right-hand sides
E(lambda) (the 'Euler characteristic', Kac's formula applied to an atypical
weight) satisfy
   E(lambda_n) = ch L(lambda_n) + ch L(lambda_{n-1})   [+ ch C at the boundary]
   E^s(lambda_n) = sch L(lambda_n) - sch L(lambda_{n-1}) [- sch C at the boundary]
Then dim/sdim follow by induction for every n."""
import charcheck as C
from charcheck import lmul, ladd, character, kac_sides
from fractions import Fraction as Fr
from itertools import product

def neg(p): return {k: -v for k, v in p.items()}

def run32(N=5):
    C.NC = 2; dbl = 2
    POSEV = [(1, 0), (0, 2)]; POSOD = [(1, -1), (0, 1), (1, 1)]
    rho = [Fr(1, 2) * (POSEV[0][i] + POSEV[1][i]) - Fr(1, 2) * sum(r[i] for r in POSOD) for i in range(2)]
    A = [[0, 1], [-1, 1]]; par = [1, 1]; SIMPLE = [(1, -1), (0, 1)]
    frm = lambda u, v: Fr(u[0]) * v[0] - Fr(u[1]) * v[1]
    W = [(1, lambda v: v), (-1, lambda v: [-v[0], v[1]]), (-1, lambda v: [v[0], -v[1]]), (1, lambda v: [-v[0], -v[1]])]
    WS = [(1, lambda v: v), (-1, lambda v: [-v[0], v[1]]), (1, lambda v: [v[0], -v[1]]), (-1, lambda v: [-v[0], -v[1]])]
    def L(n):
        lam = (Fr(n + 1), Fr(n)) if n >= 0 else (Fr(0), Fr(0))
        if n == -1: lam = (Fr(0), Fr(0))
        labels = [frm(lam, SIMPLE[0]), frm(lam, SIMPLE[1]) / Fr(-1)]
        return lam, character(A, par, labels, SIMPLE, lam, dbl)
    chs = {n: L(n) for n in range(-1, N + 1)}   # n=-1: trivial module
    ok = 0
    for n in range(1, N + 1):
        lam = chs[n][0]
        den, numP, numM, wsum = kac_sides(POSEV, POSOD, lam, rho, W, dbl)
        _, _, _, wsumS = kac_sides(POSEV, POSOD, lam, rho, WS, dbl)
        c = ladd(chs[n][1][0], chs[n - 1][1][0]); s = ladd(chs[n][1][1], neg(chs[n - 1][1][1]))
        if n == 1:
            c = ladd(c, chs[-1][1][0]); s = ladd(s, neg(chs[-1][1][1]))
        g1 = lmul(c, den) == lmul(numP, wsum)
        # supercharacter: overall sign fixed by the data
        g2 = lmul(s, den) == lmul(numM, wsumS) or lmul(s, den) == neg(lmul(numM, wsumS))
        print("osp(3|2) n=%d  ch: %s  sch: %s   dim L=%d" % (n, g1, g2, sum(chs[n][1][0].values())))
        ok += g1 and g2
    return ok

def run42(K=6, avals=(Fr(1), Fr(2), Fr(5, 3), Fr(-3))):
    C.NC = 3; dbl = 2
    POSEV = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    POSOD = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]
    rho = [-1, 1, 1]; SIMPLE = [(1, -1, -1), (0, 2, 0), (0, 0, 2)]
    W = []
    for s in product([1, -1], repeat=3):
        W.append((s[0] * s[1] * s[2], (lambda ss: (lambda v: [ss[i] * v[i] for i in range(3)]))(s)))
    ok = 0
    for a in avals:
        A = [[0, 1, a], [-1, 2, 0], [-1, 0, 2]]; par = [1, 0, 0]
        def L(k):
            m = (k + 1, k - 1, k - 1) if k >= 1 else (0, 0, 0)
            lam = tuple(Fr(x) for x in m)
            labels = [Fr(m[0] * (1 + a) + m[1] + m[2] * a) / 2, Fr(m[1]), Fr(m[2])]
            return lam, character(A, par, labels, SIMPLE, lam, dbl)
        chs = {k: L(k) for k in range(0, K + 1)}
        for k in range(2, K + 1):
            lam = chs[k][0]
            den, numP, numM, wsum = kac_sides(POSEV, POSOD, lam, rho, W, dbl)
            c = ladd(chs[k][1][0], chs[k - 1][1][0]); s = ladd(chs[k][1][1], neg(chs[k - 1][1][1]))
            if k == 2:
                c = ladd(c, chs[0][1][0]); s = ladd(s, neg(chs[0][1][1]))
            g1 = lmul(c, den) == lmul(numP, wsum)
            g2 = lmul(s, den) in (lmul(numM, wsum), neg(lmul(numM, wsum)))
            print("osp_a(4|2) a=%s k=%d  ch: %s  sch: %s  dim L=%d" % (a, k, g1, g2, sum(chs[k][1][0].values())))
            ok += g1 and g2
    return ok

if __name__ == "__main__":
    print("passed", run32(), "of 5")
    print("passed", run42(), "of", 5 * 4)
