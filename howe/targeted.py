"""targeted.py -- verify the four predicted families at a single (lambda, degree, weight)."""
import sys, time
from fractions import Fraction as F
from imod import Le, Induced
from report import render
from scan import fmtwt

def one(n, lam, m, nu):
    le = Le(n, odd_up=False)
    I = Induced(le, lam)
    t = time.time()
    sols = I.singular(m, tuple(F(x) for x in nu))
    names = {}
    return sols, [render(I, s, names) for s in sols], time.time() - t

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cases = []
    for k in range(2, n + 1):                       # family 1a
        lam = tuple([2] * (k - 1) + [1] * (n - k + 1))
        nu = tuple([2] * k + [1] * (n - k))
        cases.append(('1a k=%d' % k, lam, 1, nu))
    for a in (2, 3, 4):                             # family 1b
        lam = (a,) + (1,) * (n - 1)
        cases.append(('1b a=%d' % a, lam, 1, (a - 1,) + (1,) * (n - 1)))
    for c in (0, 1, 2):                             # family 1c
        lam = (0,) * (n - 1) + (-c,)
        cases.append(('1c c=%d' % c, lam, 1, (0,) * (n - 1) + (-c - 1,)))
    cases.append(('2', (1,) * n, 2, (1,) * n))      # family 2
    for name, lam, m, nu in cases:
        sols, txt, dt = one(n, lam, m, nu)
        print('%-8s V%s d=%d -> %s : %d singular vector(s)  (%.0fs)'
              % (name, fmtwt(tuple(F(x) for x in lam)), m, fmtwt(tuple(F(x) for x in nu)),
                 len(sols), dt))
        for s in txt:
            print('        ', s)
        sys.stdout.flush()
