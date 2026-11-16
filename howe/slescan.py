"""
slescan.py -- exhaustive search for singular vectors in I(V) over SLe(n).

A singular vector is  z in I(V)_m, m > 0, with  n_+ z = 0  and  L_j z = 0 for all j > 0.
L_1 generates L_{>=1} except for the single element  D_{xi_1...xi_n} in L_{n-2}
(see gencheck.py), which is therefore imposed separately whenever m >= n-2.

Kochetkov's weights are (n-1)-tuples  a_i = <weight, x_i xi_i - x_{i+1} xi_{i+1}>;
we compute in the eps-basis and print both.

Usage:  python3 slescan.py <n> <maxdeg> <box-hi> [height-cutoff]
        the box is over Kochetkov weights (a_1..a_{n-1}) in [0,hi]^{n-1}.
"""

import sys
from fractions import Fraction as F

from le import nullspace
from imod import Le
from sle import InducedSle, kerdelta
from pemod import mon
from scan import fmtwt, weights_of_V
from report import render


def lift(a):
    """Kochetkov weight (a_1..a_{n-1}) -> the eps-lift with lambda_n = 0"""
    n = len(a) + 1
    lam = [0] * n
    for i in range(n - 2, -1, -1):
        lam[i] = lam[i + 1] + a[i]
    return tuple(lam)


def koweight(mu):
    """eps-weight -> Kochetkov (n-1)-tuple"""
    a = tuple(mu[i] - mu[i + 1] for i in range(len(mu) - 1))
    return '(' + ','.join(str(x if x.denominator != 1 else x.numerator) for x in a) + ')'


def top(n):
    """the generating function xi_1...xi_n"""
    return mon(n, [0] * n, list(range(n)))


def scan_sle(n, a, maxdeg=4, ht=8, verbose=False):
    le = Le(n, odd_up=False)          # Kochetkov's Borel: n_+ = gl_+ + S^2
    lam = lift(a)
    I = InducedSle(le, lam)
    extra = [top(n)] if n >= 4 else []
    if not I.V.wtbasis(I.lam):
        return None, []
    wV = weights_of_V(I, ht)
    found = []
    for m in range(1, maxdeg + 1):
        cand = set()
        for u in I.umonos(m):
            w = I.uwt(u)
            for mu in wV:
                cand.add(tuple(x + y for x, y in zip(mu, w)))
        ops_extra = extra if (n >= 4 and m >= n - 2) else []
        for nu in sorted(cand, key=lambda t: -I.pe.ht(t)):
            sols = I.singular(m, nu)
            if sols and ops_extra:
                sols = _refine(I, sols, m, nu, ops_extra)
            if sols:
                found.append((m, nu, sols))
                if verbose:
                    print('   deg %d weight %s : %d singular vector(s)'
                          % (m, fmtwt(nu), len(sols)))
                    sys.stdout.flush()
    return I, found


def _refine(I, sols, m, nu, ops):
    """keep only those combinations of sols that are also killed by ops"""
    cols = list(range(len(sols)))
    eqs = {}
    for fi, f in enumerate(ops):
        for si, s in enumerate(sols):
            img = {}
            for key, c in s.items():
                for k2, c2 in I._act_one(f, key).items():
                    img[k2] = img.get(k2, 0) + c * c2
            for k2, c2 in img.items():
                eqs.setdefault((fi, k2), {})[si] = c2
    ker = nullspace(list(eqs.values()), cols)
    out = []
    for k in ker:
        v = {}
        for si, c in k.items():
            for key, c2 in sols[si].items():
                v[key] = v.get(key, 0) + c * c2
        v = {kk: vv for kk, vv in v.items() if vv}
        if v:
            out.append(v)
    return out


def kobox(n, hi):
    """all Kochetkov weights (a_1..a_{n-1}) with 0 <= a_i <= hi"""
    out = []

    def rec(i, cur):
        if i == n - 1:
            out.append(tuple(cur))
            return
        for v in range(hi + 1):
            rec(i + 1, cur + [v])

    rec(0, [])
    return out


if __name__ == '__main__':
    n = int(sys.argv[1])
    maxdeg = int(sys.argv[2])
    hi = int(sys.argv[3])
    ht = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    print('sle(%d): Kochetkov weights (a_1..a_%d) in [0,%d], degrees 1..%d, V-height cutoff %d'
          % (n, n - 1, hi, maxdeg, ht))
    for a in kobox(n, hi):
        I, res = scan_sle(n, a, maxdeg, ht)
        if not res:
            continue
        names = {}
        for m, nu, sols in res:
            for s in sols:
                print('a=%-10s lam=%-12s deg %d  wt %-12s = %-8s : %s'
                      % (str(a), fmtwt(I.lam), m, fmtwt(nu), koweight(nu),
                         render(I, s, names)))
        sys.stdout.flush()
