"""deltacheck.py -- Delta{f,g} = +-{f,Delta g} for f of degree 2 (i.e. f in L_0), g of degree 3.

This is what makes Ker Delta an L_0-submodule of L_1 and makes Delta intertwine the two
n_+-actions in Proposition 9.3.3 (the two generators).
"""
import sys
from fractions import Fraction as F
from le import abr, dx, dxi, padd, pscal
from pemod import Pe
from imod import Le

def lap(f, n):
    out = {}
    for i in range(n):
        out = padd(out, dx(dxi(f, i), i))
    return {k: v for k, v in out.items() if v}

def run(n):
    le, pe = Le(n, odd_up=False), Pe(n, odd_up=False)
    bad = plus = minus = 0
    for f in pe.basis:
        for g in le.pos1:
            lhs = lap(abr(f, g, n), n)
            rhs = abr(f, lap(g, n), n)
            if lhs == rhs:
                plus += 1
            elif lhs == pscal(rhs, -1):
                minus += 1
            else:
                bad += 1
                if bad < 4:
                    print('   MISMATCH f=%s g=%s : %s vs %s' % (f, g, lhs, rhs))
    print('n = %d : %d pairs with Delta{f,g} = +{f,Delta g}, %d with -, %d mismatches'
          % (n, plus, minus, bad))
    sys.stdout.flush()

if __name__ == '__main__':
    for n in (2, 3, 4, 5):
        run(n)
