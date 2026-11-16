"""
l1gen2.py -- how many n_+-generators does L_1 need, and which?

l1gen.py showed that no single vector generates L_1 over n_+ once n >= 3: the ht-lowest
vector xi_{n-2}xi_{n-1}xi_n generates a subspace of codimension exactly 2n = dim L_{-1},
which is Ker(Delta) for the odd Laplacian Delta = sum_i d^2/dx_i dxi_i : S^3 W -> W
(W = <x_i,xi_i>).  Here we compute the minimal number of generators,

    dim L_1 / (n_+ . L_1),

and exhibit generators of lowest ht.  Two equations, not one, then suffice for singularity.
"""

import sys
from fractions import Fraction as F
from le import abr, rref, pwt, dx, dxi, padd
from pemod import Pe
from imod import Le
from l1gen import span_dim, closure, name


def laplacian(f, n):
    out = {}
    for i in range(n):
        out = padd(out, dx(dxi(f, i), i))
    return {k: v for k, v in out.items() if v}


def run(n):
    le = Le(n, odd_up=False)
    pe = Pe(n, odd_up=False)
    L1 = le.pos1
    monos = sorted({m for f in L1 for m in f}, key=str)
    idx = {m: i for i, m in enumerate(monos)}
    dimL1 = span_dim(L1, idx)
    prod = [abr(e, f, n) for e in pe.npos for f in L1]
    dimprod = span_dim(prod, idx)
    print('n = %d :  dim L_1 = %3d,  dim n_+.L_1 = %3d,  minimal # of generators = %d'
          % (n, dimL1, dimprod, dimL1 - dimprod))

    # ht-lowest vector, and the ht-lowest partner completing it to a generating set
    hts = sorted(((pe.ht(pwt(f, n)), i) for i, f in enumerate(L1)))
    X0 = L1[hts[0][1]]
    _, d0 = closure([X0], pe.npos, n, idx)
    print('        X_0 = %-12s (ht %d) generates %d;  Delta X_0 = %s'
          % (name(n, X0), hts[0][0], d0, name(n, laplacian(X0, n)) or '0'))
    if d0 < dimL1:
        for h, i in hts:
            Y = L1[i]
            _, d = closure([X0, Y], pe.npos, n, idx)
            if d == dimL1:
                print('        + Y_0 = %-12s (ht %d):  <X_0,Y_0> generates all of L_1;'
                      '  Delta Y_0 = %s' % (name(n, Y), h, name(n, laplacian(Y, n)) or '0'))
                break
        else:
            print('        no two-element generating set found among the monomials')
    sys.stdout.flush()


if __name__ == '__main__':
    for n in (2, 3, 4, 5, 6):
        run(n)
