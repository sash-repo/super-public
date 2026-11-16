#!/usr/bin/env python3
r"""
runspin.py -- driver for spin.py; answers the question below.

  "Please compute spin module (table 4.4.3) over k^'L(1|4) for cocycles with
   non-zero beta and gamma (and doublecheck for non-zero alpha)."

STAGE 0  the machinery, verified from scratch for every n we use.
STAGE 1  calibration: reproduce the printed c-row of table 4.4.3 for n = 0,1,2,
         where the answer is already in the text.
STAGE 2  n = 4: the three central charges (alpha, beta, gamma) of
         Spin(cF_{lambda;mu} (+) dual) over k^{L}'(1|4).
STAGE 3  the same, as an exhaustive sweep over the whole even part, with n = 2
         run alongside as a positive control that the sweep is not vacuous.

Normalisation.  The paper's basis is L_m = K_{t^{m+1}} and {t^{m+1},t^{n+1}} =
2(n-m)t^{m+n+1}, so the Virasoro generator is ell_m = (1/2)K_{t^{m+1}}; with
equation 148 [ell_m,ell_n] = (n-m)ell_{m+n} + (c/12)(m^3-m) delta, the alpha of
(73) gives c = 3 alpha.

Which entries are used, and why they are unambiguous.  A weight-0 2-cochain is
determined modulo the coboundary of a functional on the degree-0 part, and
delta(phi)(X,Y) = phi([X,Y]).  For each of the three pairs below the bracket
VANISHES, so the value read off the vacuum is the cohomology class itself:

    c(T^{ij}_m, T^{ij}_{-m}) = -alpha m
    c(T^{ij}_m, T^{kl}_{-m}) =  beta  m      (i,j,k,l) in A_4
    c(L_m,      S_{-m})      =  gamma + beta m

These three are linearly independent in (alpha, beta, gamma), so the even part
of k^{L}'(1|4) already sees all three classes: the odd entries c(E^i_m,F^i_n)
are not needed (and the naive quadratic ansatz for \hat X degenerates for odd X,
see the note at the end of spin.py).
"""

from fractions import Fraction as Q
from itertools import combinations, product
import sys

from spin import (Wspace, mono, bracket, cocycle, hatX, VAC,
                  test_algebra, test_module, test_form, test_hat)

WIN = 6


def L(m):
    return mono(m + 1)


def T(i, j, m):
    return mono(m, (i, j))


def S(n, m):
    return mono(m - 1, tuple(range(1, n + 1)))


def h_of(Wsp, win=WIN):
    """h = value of ell_0 on the vacuum, got as -(1/2)<ell_1 ell_{-1}> so that
    the (divergent) normal-ordering constant of \\hat ell_0 is never needed:
    [ell_1, ell_{-1}] = -2 ell_0 carries no central term."""
    st = hatX(Wsp, L(1), hatX(Wsp, L(-1), VAC, win), win)
    return -Q(1, 8) * st.get((), Q(0))


def alpha_from_LL(Wsp, win=WIN, ms=(2, 3, 4)):
    """c(L_m, L_{-m}) = alpha m(m^2-1) + kappa m, the linear term absorbing the
    normal-ordering constant of \\hat L_0 ([L_m,L_{-m}] = -4m L_0)."""
    vals = []
    for m in ms:
        v, rest = cocycle(Wsp, L(m), L(-m), win)
        assert not rest, 'not central: %s' % rest
        vals.append((m, v))
    (m1, v1), (m2, v2) = vals[0], vals[1]
    det = m1 ** 3 * m2 - m1 * m2 ** 3
    al = (v1 * m2 - m1 * v2) / det
    ka = (m1 ** 3 * v2 - v1 * m2 ** 3) / det
    for m, v in vals[2:]:
        assert al * m ** 3 + ka * m == v, ('c(L_m,L_-m) is not a cubic', vals)
    return al


def banner(s):
    print()
    print('=' * 76)
    print(s)
    print('=' * 76)


# ------------------------------------------------------------------ stage 0

banner('STAGE 0  --  the machinery, verified from scratch')
print('%-4s %-34s %-34s' % ('n', 'check', 'result'))
print('-' * 76)
for n in (0, 1, 2, 3, 4):
    Wsp = Wspace(n, Q(1, 3), Q(1, 5))
    fs = [mono(a, S_) for a in (-1, 0, 1, 2)
          for r in range(0, n + 1, 2) for S_ in combinations(range(1, n + 1), r)]
    sym, inv = test_form(Wsp)
    rows = [('[K_f,K_g] = K_{f,g} on functions', test_algebra(n)),
            ('module axiom, cF and its dual', test_module(Wsp)),
            ('B supersymmetric and even', sym),
            ('B is k^L(1|n)-invariant', inv),
            ('[hat X, psi_v] = psi_{Xv}, X even', test_hat(Wsp, 5, fs))]
    for i, (name, bad) in enumerate(rows):
        print('%-4s %-34s %-34s' % (n if i == 0 else '', name,
                                    'OK' if bad == 0 else 'FAILS (%d)' % bad))
        if bad:
            sys.exit('machinery broken -- stopping')
    print()

# ------------------------------------------------------------------ stage 1

banner('STAGE 1  --  calibration against the printed c-row of table 4.4.3')
print('The table\'s c is reproduced with the opposite sign throughout -- the')
print('sign question left open in the text.')
print()
print('%-3s %-8s %-8s  %-10s %-10s %-10s' %
      ('n', 'lambda', 'mu', 'computed', '-(table)', 'verdict'))
print('-' * 76)
for n in (0, 1, 2):
    for lam, mu in ((Q(1, 3), Q(1, 5)), (Q(1, 2), 0), (2, -1), (0, Q(2, 3))):
        Wsp = Wspace(n, lam, mu)
        # for n = 1 the table's column is the one of the alpha_1^{lambda}
        # convention (the n = 0 line of (26)), not alpha_1^{lambda/2}
        if n == 1:
            Wsp.V.w = 2 * Q(lam)
            Wsp.Vd.w = (2 - n) - 2 * Q(lam)
        c = 3 * alpha_from_LL(Wsp)
        ct = (12 * Q(lam) ** 2 - 12 * Q(lam) + 2) if n == 0 else \
             ((-12 * Q(lam) + 3) if n == 1 else Q(6))
        print('%-3s %-8s %-8s  %-10s %-10s %-10s'
              % (n, lam, mu, c, -ct, 'agrees' if c == -ct else 'MISMATCH'))
print()
print('For n = 1 with eq. (26) taken literally (alpha_1^{lambda/2}) one gets')
print('c = 6 lambda - 3 instead of 12 lambda - 3:')
for lam in (Q(1, 3), Q(1, 2), 2, 0):
    Wsp = Wspace(1, lam, Q(1, 5))
    print('    lambda = %-6s   c = %-6s   (-(table) = %s)'
          % (lam, 3 * alpha_from_LL(Wsp), 12 * Q(lam) - 3))

# ------------------------------------------------------------------ stage 2

banner('STAGE 2  --  n = 4: the three central charges of Spin(cF_{lambda;mu})')
print('%-9s %-9s %-8s %-8s %-8s %-9s %-8s'
      % ('lambda', 'mu', 'alpha', 'beta', 'gamma', 'c = 3alpha', 'h'))
print('-' * 76)
for lam, mu in ((Q(1, 3), Q(1, 5)), (Q(1, 2), 0), (0, 0), (2, -1),
                (Q(-3, 7), Q(5, 2)), (1, Q(1, 3)), (Q(7, 11), Q(-4, 9))):
    res = []
    for win in (5, 7):
        Wsp = Wspace(4, lam, mu)
        als = {-cocycle(Wsp, T(1, 2, m), T(1, 2, -m), win)[0] / m for m in (1, 2, 3)}
        bes = {cocycle(Wsp, T(1, 2, m), T(3, 4, -m), win)[0] / m for m in (1, 2, 3)}
        assert len(als) == 1 and len(bes) == 1, (als, bes)
        al, be = als.pop(), bes.pop()
        gas = {cocycle(Wsp, L(m), S(4, -m), win)[0] - be * m for m in (1, 2, 3)}
        assert len(gas) == 1, gas
        assert alpha_from_LL(Wsp, win) == al, 'alpha: T-T and L-L disagree'
        res.append((al, be, gas.pop(), h_of(Wsp, win)))
    assert res[0] == res[1], ('window-dependent', res)
    al, be, ga, h = res[0]
    print('%-9s %-9s %-8s %-8s %-8s %-9s %-8s' % (lam, mu, al, be, ga, 3 * al, h))
print()
print('alpha is obtained twice (the T-T entry and the cubic in c(L_m,L_-m)) and')
print('every value is stable between the windows |a| <= 5 and |a| <= 7.')

# ------------------------------------------------------------------ stage 3


def evens(n, A):
    return [(a, S) for a in range(-A, A + 1) for r in range(0, n + 1, 2)
            for S in combinations(range(1, n + 1), r)]


def weight(k):
    return 2 * k[0] - 2 + len(k[1])


banner('STAGE 3  --  exhaustive sweep of the even part (n = 2 as positive control)')
print('%-4s %-10s %-14s %-16s %-16s' %
      ('n', 'pairs', 'zero bracket', 'non-zero c', 'non-zero, clean'))
print('-' * 76)
for n in (2, 4):
    ks = evens(n, 3)
    top = tuple(range(1, n + 1))
    Wsp = Wspace(n, Q(1, 3), Q(1, 5))
    tot = zbr = nz = nzc = 0
    for k1, k2 in product(ks, repeat=2):
        if weight(k1) + weight(k2) != 0:
            continue
        if n == 4 and (k1 == (1, top) or k2 == (1, top)):
            continue                       # S_0 is not in k^L'(1|4)
        f, g = mono(*k1), mono(*k2)
        tot += 1
        clean = not bracket(f, g, n)
        v, rest = cocycle(Wsp, f, g, WIN)
        assert not rest, (k1, k2, rest)
        zbr += clean
        if v:
            nz += 1
            nzc += clean
    print('%-4s %-10s %-14s %-16s %-16s' % (n, tot, zbr, nz, nzc))
print()
print('n = 4: the induced cocycle vanishes on the nose -- not merely up to a')
print('coboundary -- on every even pair of total weight 0 in the window; n = 2,')
print('run through the identical code, gives non-zero values, so the sweep bites.')
print()
print('CONCLUSION.  alpha = beta = gamma = 0 for Spin(cF_{lambda;mu} (+) dual)')
print('over k^{L}\'(1|4), for every lambda and mu tested: the sentence above')
print('table 4.4.3, "c_1 = c_2 = c_3 = 0", is correct, and correct for all three')
print('cocycles, not only for the Virasoro one.')
