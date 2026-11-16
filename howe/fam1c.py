"""
fam1c.py -- family 1c) through the two equations of Subsec. "Two equations instead of all of L_1".

For z = delta_n (x) m_1 (so v_i = 0 for all i, w_j = 0 for j < n, w_n = m_1) the two equations
    (E1)  b_{n-1,n} w_{n-2} - b_{n-2,n} w_{n-1} + b_{n-2,n-1} w_n = 0
    (E2)  b_{n-1,n} v_{n-1} - a_{n-1,n} w_{n-1} + e_{n-1} w_n     = 0
read
    (E1)  b_{n-2,n-1} . m_1 = 0,      (E2)  e_{n-1} . m_1 = 0, i.e. lambda_{n-1} = 0.
Together with n_+ z = 0 that is the whole of the singularity of delta_n (x) m_1.
Here we check both against the engine for lambda = (0,...,0,-c).
"""

import sys
from fractions import Fraction as F
from pemod import Pe, Irrep, xixi
from imod import Le


def run(n, c):
    pe = Pe(n, odd_up=False)
    lam = tuple([0] * (n - 1) + [-c])
    V = Irrep(pe, lam)
    m1 = {0: F(1)}
    k = next(i for i, f in enumerate(pe.basis) if f == xixi(n, n - 3, n - 2)) if n >= 3 else None
    out = 'n=%d c=%d  lambda=%s :  lambda_{n-1} = %d' % (n, c, str(lam), lam[n - 2])
    if k is not None:
        img = V.act(k, lam, m1)
        out += ',   b_{%d%d}.m_1 = %s' % (n - 2, n - 1, '0' if not img else str(img))
    print(out)
    sys.stdout.flush()


if __name__ == '__main__':
    for n in (3, 4, 5):
        for c in (0, 1, 2, 3):
            run(n, c)
