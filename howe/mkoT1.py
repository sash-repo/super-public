"""
mkoT1.py -- Kochetkov's Theorem 1 (singular vectors for M(n)), family by family,
at a given n.  Each family is tested at its own predicted weight: we print the whole
space of singular vectors of that degree and weight, so that both existence and
uniqueness are checked, and we say whether the module V is finite-dimensional
(= lambda dominant for pe(n)) or not.

usage:  python3 mkoT1.py <n>
"""

import sys
from fractions import Fraction as F

from mimod import Mn, MInduced
from mreport import render


def dom(lam):
    return all(lam[i] >= lam[i + 1] for i in range(len(lam) - 1))


def at(mn, lam, c, m, nu, note="", ops=None):
    n = mn.n
    I = MInduced(mn, lam, c, ops)
    try:
        sv = I.singular(m, tuple(F(x) for x in nu))
    except Exception as e:
        print("    ERROR %s" % e)
        return []
    d = "dominant" if dom(lam) else "NOT dominant (dim V = infinity)"
    print("  V(%s), z=%s [%s] ;  m=%d, wt(y)=(%s; %s) : %d singular vector(s) %s"
          % (",".join(str(x) for x in lam), c, d, m,
             ",".join(str(x) for x in nu), F(c) - m, len(sv), note))
    for s in sv:
        print("      " + render(I, s))
    return sv


def main(n):
    mn = Mn(n)
    print("=== Kochetkov's Theorem 1 (M(%d)) ===" % n)
    print("-- 1a)  v_k h.w. (2^{k-1},1^{n-k+1}; n-k+3),  wt y = (2^k,1^{n-k}; n-k+2)")
    for k in range(1, n + 1):
        lam = [2] * (k - 1) + [1] * (n - k + 1)
        c = n - k + 3
        nu = [2] * k + [1] * (n - k)
        at(mn, lam, c, 1, nu, "(k=%d)" % k)
    print("-- 1b)  w_k h.w. (0^{k-1},lam+1,1^{n-k}; n-k+lam+1),  wt y = (0^{k-1},lam,1^{n-k}; n-k+lam)")
    for k in range(1, n + 1):
        for lm in [-2, -1, 0, 1, 2]:
            lam = [0] * (k - 1) + [lm + 1] + [1] * (n - k)
            c = n - k + lm + 1
            nu = [0] * (k - 1) + [lm] + [1] * (n - k)
            at(mn, lam, c, 1, nu, "(k=%d, lambda=%d)" % (k, lm))
    print("-- 2a)  v h.w. (1,...,1; lam+2),  wt y = (1,...,1; lam),  w = -(n+lam) v")
    for lm in [-3, -2, -1, 0, 1, 2]:
        at(mn, [1] * n, lm + 2, 2, [1] * n, "(lambda=%d)" % lm)
    if n == 2:
        print("-- 2b)  n=2:  y = 'd1'd2 (x) v,  v h.w. (1,1;2),  wt y = (0,0;0)")
        at(mn, [1, 1], 2, 2, [0, 0])


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
