#!/usr/bin/env python3
"""An independent check of the five-sector classification of Section 8:
pe(2)-invariant BINARY differential operators F_{l1,n1} (x) F_{l2,n2} -> F_{mu,om}
over m(1), sector by sector.

Sectors.  Every binary monomial is  P1 (x) P2  with  P = D_xi^a D_q^e d_tau^d,
   a >= 0,  e, d in {0,1}      (the normal form of Lemma 7.7),
   k  = a1+e1+2d1 + a2+e2+2d2                (the degree),
   u  = a1+a2+d1+d2,   v = e1+e2+d1+d2,      u+v = k,
and the parity of the monomial is v mod 2.  The five sectors are v=0,...,4:
   odd   A: v=1,   odd   B: v=3,
   even  A: v=0,   even  B: v=2,   even C: v=4.
The target weights are mu = l1+l2-u, om = n1+n2+(u-v)/2.

What is computed here is dim Inv^{(k)}_{v}(l1,l2), i.e., the dimension of the
space of invariant operators in the sector, either at symbolic weights (the
generic dimension) or at a numerical pair (l1,l2); the weights n1, n2 stay
symbolic throughout, since nothing depends on them.

Equations are taken from M_{q^2}, M_{q tau}, M_{tau xi}; every solution is then
re-verified against all eight generating functions of pe(2).  This is the same
protocol as classify_m1.py, whose eqs_bin is reused.
"""
import sys
from fractions import Fraction
from sympy import Matrix, S, expand, factor, simplify, symbols, Rational

from super12 import mono, compose
from m1 import D_1, D_xi, D_q, PE
from classify_m1 import eqs_bin, FIELDS, ALLF

l1, n1, l2, n2 = symbols('lambda_1 nu_1 lambda_2 nu_2')

NAMES = {0: 'even A', 1: 'odd  A', 2: 'even B', 3: 'odd  B', 4: 'even C'}


def P(a, e, d):
    """D_xi^a D_q^e d_tau^d  (d_tau acts first)."""
    ops = [D_xi] * a + [D_q] * e + [D_1] * d
    return compose(*ops) if ops else (lambda u: dict(u))


def pname(a, e, d):
    s = ''
    if a:
        s += 'X' + (str(a) if a > 1 else '')
    if e:
        s += 'Q'
    if d:
        s += 'T'
    return s or '1'


def terms_of(k, v):
    """the monomials of the sector (k, v), in the normal form of Section 8 of the paper."""
    out = []
    for a1_ in range(k + 1):
        for e1_ in (0, 1):
            for d1_ in (0, 1):
                r = k - (a1_ + e1_ + 2 * d1_)
                if r < 0:
                    continue
                for e2_ in (0, 1):
                    for d2_ in (0, 1):
                        a2_ = r - e2_ - 2 * d2_
                        if a2_ < 0:
                            continue
                        if e1_ + e2_ + d1_ + d2_ != v:
                            continue
                        out.append((f"{pname(a1_, e1_, d1_)}|{pname(a2_, e2_, d2_)}",
                                    P(a1_, e1_, d1_), P(a2_, e2_, d2_),
                                    (e2_ + d2_) % 2))
    return out


def dim_sector(k, v, L1=l1, L2=l2, nmax=None, fields=FIELDS, verify=False):
    """dim of the space of invariant operators in the sector (k,v)."""
    tt = terms_of(k, v)
    if not tt:
        return 0, [], []
    u = k - v
    At = L1 + L2 - u
    Bt = n1 + n2 + S(u - v) / 2
    if nmax is None:
        nmax = k + 1
    cs, eqs = eqs_bin(tt, L1, n1, L2, n2, At, Bt, v % 2, nmax, fields)
    Mx = Matrix([[expand(e).coeff(c) for c in cs] for e in eqs])
    if Mx.rows == 0:
        ns = [Matrix([1 if i == j else 0 for i in range(len(cs))])
              for j in range(len(cs))]
    else:
        ns = Mx.nullspace()
    if verify and ns:
        # re-verify every basis vector against all eight generating functions
        from sympy import cancel, together
        _, eqs2 = eqs_bin(tt, L1, n1, L2, n2, At, Bt, v % 2, nmax, ALLF)
        for vec in ns:
            sub = {cs[i]: vec[i] for i in range(len(cs))}
            bad = [e for e in eqs2 if cancel(together(e.subs(sub))) != 0]
            assert not bad, (f"k={k} v={v}: solution fails the full check on "
                             f"{len(bad)} of {len(eqs2)} equations")
    return len(ns), tt, ns


def show(k, v, ns, tt):
    for vec in ns:
        d = {tt[i][0]: factor(simplify(vec[i]))
             for i in range(len(tt)) if vec[i] != 0}
        print("       ", d, flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'generic'
    KMAX = int(sys.argv[2]) if (len(sys.argv) > 2 and mode != 'grid') else 6

    if mode == 'num':
        # "generic" dimension read off at a non-resonant numerical pair, which
        # is cheap enough to reach the degrees the symbolic run cannot.
        L1 = Rational(int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 \
            else Rational(7, 3)
        L2 = Rational(int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 \
            else Rational(11, 5)
        print(f"NUMERICAL weights lambda_1={L1}, lambda_2={L2}: dim Inv^{{(k)}}_v")
        for k in range(1, KMAX + 1):
            for v in range(0, 5):
                if not terms_of(k, v):
                    continue
                d, tt, ns = dim_sector(k, v, L1, L2)
                print(f"  k={k}  v={v} ({NAMES[v]}): {len(tt):3d} terms, dim = {d}",
                      flush=True)

    elif mode == 'generic':
        print("=" * 78)
        print("GENERIC weights (lambda_1, lambda_2 symbolic): dim Inv^{(k)}_v")
        print("=" * 78)
        for k in range(1, KMAX + 1):
            for v in range(0, 5):
                tt = terms_of(k, v)
                if not tt:
                    continue
                d, tt, ns = dim_sector(k, v, verify=True)
                print(f"  k={k}  v={v} ({NAMES[v]}): {len(tt):3d} terms, "
                      f"generic dim = {d}", flush=True)
                if d and k <= 4:
                    show(k, v, ns, tt)

    elif mode == 'grid':
        # resonance scan: the grid is in 2*lambda, which runs through integers
        KLIST = [int(x) for x in sys.argv[2].split(',')]
        v = int(sys.argv[3])
        LO = Rational(int(sys.argv[4]) if len(sys.argv) > 4 else 0, 1)
        HI = Rational(int(sys.argv[5]) if len(sys.argv) > 5 else 5, 1)
        print(f"GRID sector v={v} ({NAMES[v]}), 2*lambda in "
              f"[{2*LO},{2*HI}] by halves, k in {KLIST}")
        for k in KLIST:
            if not terms_of(k, v):
                continue
            gen, _, _ = dim_sector(k, v, Rational(7, 3), Rational(11, 5))
            print(f"  --- k={k}: generic dim {gen}", flush=True)
            t2 = 2 * LO
            while t2 <= 2 * HI:
                s2 = 2 * LO
                while s2 <= 2 * HI:
                    d, tt, ns = dim_sector(k, v, Rational(t2, 2), Rational(s2, 2))
                    if d != gen:
                        print(f"      2l1={t2} 2l2={s2}: dim {d}", flush=True)
                    s2 += 1
                t2 += 1

    elif mode == 'pts':
        # a targeted check of a resonance condition: the points are given as
        # k:2l1:2l2 and the dimension is printed for each.
        v = int(sys.argv[3])
        for spec in sys.argv[4].split(','):
            k, t2, s2 = (int(x) for x in spec.split(':'))
            d, tt, ns = dim_sector(k, v, Rational(t2, 2), Rational(s2, 2))
            print(f"  v={v} k={k}  2l1={t2:3d} 2l2={s2:3d}: dim = {d}", flush=True)
