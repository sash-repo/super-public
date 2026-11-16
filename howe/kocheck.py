"""
kocheck.py -- Kochetkov's nine families of singular vectors for SLe(n)
(УМН 43 (1988), no. 2, 201-202, "Особые векторы и инвариантные операторы"),
checked one by one against the engine.

His weights are (n-1)-tuples in the basis dual to  h_i = x_i xi_i - x_{i+1} xi_{i+1};
omega_j = "1 in the j-th place", with the convention omega_0 = omega_n = 0.

    1a)  V = V(omega_{k-1}),  2 <= k <= n,   m = 1,  wt z = omega_k
    1b)  V = V(lam*omega_1),  lam >= 1,      m = 1,  wt z = (lam-1) omega_1
    1c)  V = V(lam*omega_{n-1}), lam >= 0,   m = 1,  wt z = (lam+1) omega_{n-1}
    2a)  V = V(0),                           m = 2,  wt z = 0
    2b)  V = V(omega_{n-1}),                 m = 2,  wt z = omega_{n-1}
    2c)  V = V(2 omega_{n-1}),               m = 2,  wt z = omega_{n-2}
    3a)  V = V(omega_{n-1}),                 m = 3,  wt z = 0
    3b)  V = V(0),                           m = 3,  wt z = omega_{n-1}
    4)   V = V(omega_{n-1}),                 m = 4,  wt z = omega_{n-1}

Usage:  python3 kocheck.py <n> [lam-max]
"""

import sys
from fractions import Fraction as F

from imod import Le
from sle import InducedSle
from slescan import lift, koweight, top, scan_sle, _refine
from scan import fmtwt, weights_of_V
from report import render


def om(n, j):
    """omega_j as a Kochetkov (n-1)-tuple; omega_0 = omega_n = 0"""
    a = [0] * (n - 1)
    if 1 <= j <= n - 1:
        a[j - 1] = 1
    return tuple(a)


def scal(a, c):
    return tuple(c * x for x in a)


def families(n, lmax=3):
    out = []
    for k in range(2, n + 1):
        out.append(('1a k=%d' % k, om(n, k - 1), 1, om(n, k)))
    for lam in range(1, lmax + 1):
        out.append(('1b lam=%d' % lam, scal(om(n, 1), lam), 1, scal(om(n, 1), lam - 1)))
    for lam in range(0, lmax + 1):
        out.append(('1c lam=%d' % lam, scal(om(n, n - 1), lam), 1, scal(om(n, n - 1), lam + 1)))
    out.append(('2a', om(n, 0), 2, om(n, 0)))
    out.append(('2b', om(n, n - 1), 2, om(n, n - 1)))
    out.append(('2c printed', scal(om(n, n - 1), 2), 2, om(n, n - 2)))
    out.append(('2c swapped', om(n, n - 2), 2, scal(om(n, n - 1), 2)))
    out.append(('3a', om(n, n - 1), 3, om(n, 0)))
    out.append(('3b', om(n, 0), 3, om(n, n - 1)))
    out.append(('4', om(n, n - 1), 4, om(n, n - 1)))
    return out


def check(n, lmax=3, ht=8):
    le = Le(n, odd_up=False)
    cache = {}
    for name, a, m, b in families(n, lmax):
        lam = lift(a)
        if lam not in cache:
            cache[lam] = (InducedSle(le, lam), {})
        I, names = cache[lam]
        if not I.V.wtbasis(I.lam):
            print('%-10s V=%-10s dim V = 0 (no such module)' % (name, str(a)))
            continue
        wV = weights_of_V(I, ht)
        cand = set()
        for u in I.umonos(m):
            w = I.uwt(u)
            for mu in wV:
                cand.add(tuple(x + y for x, y in zip(mu, w)))
        target = '(' + ','.join(str(x) for x in b) + ')'
        hits = [nu for nu in cand if koweight(nu) == target]
        tot = 0
        lines = []
        for nu in sorted(hits, key=lambda t: -I.pe.ht(t)):
            sols = I.singular(m, nu)
            if sols and n >= 4 and m >= n - 2:
                sols = _refine(I, sols, m, nu, [top(n)])
            for s in sols:
                tot += 1
                lines.append('      eps-wt %-14s : %s' % (fmtwt(nu), render(I, s, names)))
        print('%-12s V=%-10s m=%d  wt z=%-10s : %d singular vector(s)'
              % (name, str(a), m, str(b), tot))
        for ln in lines:
            print(ln)
        sys.stdout.flush()


if __name__ == '__main__':
    n = int(sys.argv[1])
    lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ht = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    print("SLe(%d): Kochetkov's families checked one by one (Borel: S^2 up)" % n)
    check(n, lmax, ht)
