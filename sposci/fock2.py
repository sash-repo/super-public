#!/usr/bin/env python3
r"""
fock2.py -- same computation as fock.py, but for the *doubled* module
W = F_{lambda, mu} (+) F_{1-lambda, -mu}, the case treated in the tables of the
section "Problems" of the text (there the module F_{lambda; mu} is arbitrary, so it
carries no invariant form and has to be doubled, W = V (+) V^*).

Purpose: to see which normalisation the tables use, and hence whether the four
values of the Lemma are meant for the self-dual modules of Statement 4.5 (then
they are the numbers of fock.py) or for their doubles (then they are twice as
big).  The table for k^L(1|n) claims, for n = 0, i.e. for vir itself,

      c = 12 lambda^2 - 12 lambda + 2,     h = (mu + 2 lambda)(mu + 1).

Modes are pairs (s, i), s = 0 for phi_i = t^{i+mu}(dt)^lambda and s = 1 for
psi_j = t^{j-mu}(dt)^{1-lambda}; the pairing is B(phi_i, psi_j) = Res phi_i psi_j
= delta_{i+j, -1} and B(phi, phi) = B(psi, psi) = 0, and

  L_a phi_i = (mu + i + lambda (a+1)) phi_{i+a},
  L_a psi_j = (-mu + j + (1-lambda)(a+1)) psi_{j+a}.

Polarisation: creation = {phi_i, psi_i | i <= -1}, annihilation = {i >= 0};
both halves are isotropic, so this is the standard vacuum.  Everything else
(the 1/2 in hat X = (1/2) sum X(e_a) e^a, the checks, the extraction of c and h)
is as in fock.py.
"""

from fractions import Fraction as F
from itertools import product

W = 30

ONE = {((), 0): F(1)}


class Doubled:
    parity = 'odd'          # spinor: Clifford

    def __init__(self, lam, mu):
        self.lam, self.mu = F(lam), F(mu)

    def B(self, m, n):
        (s, i), (s2, j) = m, n
        if s == s2:
            return F(0)
        return F(1) if i + j == -1 else F(0)

    def sigma(self, m):
        s, i = m
        return (1 - s, -i - 1)

    def beta(self, m):
        return self.B(self.sigma(m), m)

    def kappa(self, m, a):
        s, i = m
        return (self.mu + i + self.lam * (a + 1)) if s == 0 else \
               (-self.mu + i + (1 - self.lam) * (a + 1))

    def kind(self, m):
        return 'C' if m[1] <= -1 else 'A'

    def modes(self, lo, hi):
        return [(s, i) for s in (0, 1) for i in range(lo, hi)]


def sadd(a, b, s=F(1)):
    r = dict(a)
    for k, v in b.items():
        r[k] = r.get(k, F(0)) + s * v
        if r[k] == 0:
            del r[k]
    return r


def create(case, g, st):
    out = {}
    for (mono, th), co in st.items():
        if g in mono:
            continue
        k = sum(1 for x in mono if x < g)
        new = tuple(sorted(mono + (g,)))
        out[(new, th)] = out.get((new, th), F(0)) + (-1 if k % 2 else 1) * co
    return {k: v for k, v in out.items() if v != 0}


def annih(case, g, coef, st):
    out = {}
    for (mono, th), co in st.items():
        if g not in mono:
            continue
        k = mono.index(g)
        new = mono[:k] + mono[k + 1:]
        out[(new, th)] = out.get((new, th), F(0)) + (-1 if k % 2 else 1) * coef * co
    return {k: v for k, v in out.items() if v != 0}


def phi(case, m, st):
    if case.kind(m) == 'C':
        return create(case, m, st)
    return annih(case, case.sigma(m), case.beta(m), st)


def L(case, a, st, half=True, window=W):
    tot = {}
    for m in case.modes(-window, window + 1):
        b = case.beta(m)
        if b == 0:
            continue
        ka = case.kappa(m, a)
        if ka == 0:
            continue
        s, i = m
        tgt = (s, i + a)
        part = phi(case, tgt, phi(case, case.sigma(m), st))
        if part:
            tot = sadd(tot, part, (F(1, 2) if half else F(1)) * ka / b)
    return tot


def coeff_of_one(st):
    return st.get(((), 0), F(0))


def states(case, nmax=2):
    creators = [(s, i) for s in (0, 1) for i in (-1, -2)]
    out = [ONE]
    for r in range(1, nmax + 1):
        for combo in product(creators, repeat=r):
            st = ONE
            for g in combo:
                st = create(case, g, st)
            if st:
                out.append(st)
    seen, uniq = set(), []
    for st in out:
        k = tuple(sorted((m, str(v)) for m, v in st.items()))
        if k not in seen:
            seen.add(k)
            uniq.append(st)
    return uniq[:10]


def check_action(case):
    bad = 0
    for a in (-2, -1, 1, 2):
        for m in case.modes(-3, 3):
            for st in states(case, 1):
                lhs = sadd(L(case, a, phi(case, m, st)),
                           phi(case, m, L(case, a, st)), F(-1))
                s, i = m
                if sadd(lhs, phi(case, (s, i + a), st), -case.kappa(m, a)):
                    bad += 1
    return bad


def check_vir(case, half=True):
    bad = 0
    for a, b in ((1, 2), (-1, 2), (-1, -2), (1, 3)):
        for st in states(case, 2):
            lhs = sadd(L(case, a, L(case, b, st, half), half),
                       L(case, b, L(case, a, st, half), half), F(-1))
            if sadd(lhs, L(case, a + b, st, half), -(b - a)):
                bad += 1
    return bad


def ch(case):
    for a in (1, 2, 3):
        assert not L(case, a, ONE), 'L_%d One != 0' % a
    h = -F(1, 2) * coeff_of_one(L(case, 1, L(case, -1, ONE)))
    cs = []
    for a in (2, 3):
        v = coeff_of_one(L(case, a, L(case, -a, ONE)))
        cs.append((v + 2 * a * h) * 12 / F(a ** 3 - a))
    assert cs[0] == cs[1], cs
    return cs[0], h


if __name__ == '__main__':
    print('doubled module  W = F_{lambda,mu} (+) F_{1-lambda,-mu},  spinor rep')
    print()
    print('%-14s %-8s %-8s %-26s %-26s' %
          ('lambda', 'mu', 'checks', 'computed (c, h)', 'table row n=0'))
    print('-' * 92)
    rows = []
    for lam, mu in ((F(1, 2), 0), (F(1, 2), F(1, 2)), (0, 0), (0, F(1, 2)),
                    (F(1, 3), F(1, 5)), (2, -1), (-1, F(2, 3))):
        case = Doubled(lam, mu)
        bad = check_action(case) + check_vir(case)
        c, h = ch(case)
        ctab = 12 * F(lam) ** 2 - 12 * F(lam) + 2
        htab = (F(mu) + 2 * F(lam)) * (F(mu) + 1)
        rows.append((lam, mu, c, h, ctab, htab))
        print('%-14s %-8s %-8s (%-11s %-11s) (%-11s %-11s)'
              % (lam, mu, bad, str(c) + ',', str(h), str(ctab) + ',', str(htab)))
    print()
    print('c_computed + c_table  =', [str(r[2] + r[4]) for r in rows],
          '  <-- 0 means the table has the opposite sign')
    print('h_computed + h_table  =', [str(r[3] + r[5]) for r in rows])
    print('c_computed / c_table  =', [str(r[2] / r[4]) if r[4] else '-' for r in rows])
