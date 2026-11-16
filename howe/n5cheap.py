"""n5cheap.py -- families 1b), 1c), 2) at n=5 (the cheap half of targeted.py 5)."""
import sys
from fractions import Fraction as F
from targeted import one
from scan import fmtwt
n = 5
cases = [('1b a=%d' % a, (a,) + (1,) * (n - 1), 1, (a - 1,) + (1,) * (n - 1)) for a in (2, 3)]
cases += [('1c c=%d' % c, (0,) * (n - 1) + (-c,), 1, (0,) * (n - 1) + (-c - 1,)) for c in (0, 1, 2)]
cases += [('2', (1,) * n, 2, (1,) * n)]
for name, lam, m, nu in cases:
    sols, txt, dt = one(n, lam, m, nu)
    print('%-8s V%s d=%d -> %s : %d singular vector(s)  (%.0fs)'
          % (name, fmtwt(tuple(F(x) for x in lam)), m, fmtwt(tuple(F(x) for x in nu)), len(sols), dt))
    for s in txt:
        print('        ', s)
    sys.stdout.flush()
