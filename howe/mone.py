"""
mone.py -- scan one module I(V(lambda) (x) C_c) over M(n) (or M_{a,b}(n)) for singular
vectors of degree m, over every weight that can occur.

usage:  python3 mone.py <n> <mmax> <lam_1> ... <lam_n> [-- a b]
"""

import sys
from fractions import Fraction as F

from mn import layer, mdeg
from mimod import Mn, Mab, MInduced
from mreport import render


def weights_of(V, pe):
    seen, stack = set(), [V.lam]
    while stack:
        mu = stack.pop()
        if mu in seen:
            continue
        seen.add(mu)
        if not V.wtbasis(mu):
            continue
        for w in pe.nneg_wt:
            stack.append(tuple(a + c for a, c in zip(mu, w)))
    return [mu for mu in seen if V.wtbasis(mu)]


def scan(n, lam, zc, mmax, ops=None, tag=""):
    mn = Mn(n)
    I = MInduced(mn, lam, zc, ops)
    ws = weights_of(I.V, mn.pe)
    dimV = sum(len(I.V.wtbasis(mu)) for mu in ws)
    print("V(%s), z = %s : dim V = %d, %d weights   %s" % (list(lam), zc, dimV, len(ws), tag))
    found = []
    for m in range(1, mmax + 1):
        nus = set()
        for u in I.umonos(m):
            w = I.uwt(u)
            for mu in ws:
                nus.add(tuple(a + b for a, b in zip(mu, w)))
        for nu in sorted(nus):
            sv = I.singular(m, nu)
            for s in sv:
                found.append((m, nu, s))
                print("  m = %d  wt(y) = (%s; %s)   %s"
                      % (m, ",".join(str(x) for x in nu), I.zwt(m), render(I, s)))
    if not found:
        print("  (none)")
    return found


if __name__ == '__main__':
    args = sys.argv[1:]
    ab = None
    if '--' in args:
        i = args.index('--')
        ab = (F(args[i + 1]), F(args[i + 2]))
        args = args[:i]
    n = int(args[0])
    mmax = int(args[1])
    lam = [F(x) for x in args[2:2 + n]]
    zc = F(args[2 + n]) if len(args) > 2 + n else 0
    ops = None
    tag = ""
    if ab is not None:
        mn = Mn(n)
        A = Mab(mn, ab[0], ab[1])
        ops = A.layer(1)
        # if L_1 does not generate L_{>=1}, add L_2
        from mgencheck import span_dim
        amb = [next(iter(f)) for f in layer(n, 2)]
        gen = [x for f in A.layer(1) for x in [__import__('mn').mbr(f, g, n)]
               for g in A.layer(1) if x and mdeg(x, n) == 2]
        if span_dim(gen, amb) < len(A.layer(2)):
            ops = ops + A.layer(2)
            tag = "(L_1 + L_2 imposed)"
        tag = "M_{%s,%s}(n) " % (ab[0], ab[1]) + tag
    scan(n, lam, zc, mmax, ops, tag)
