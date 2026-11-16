#!/usr/bin/env python3
r"""
target.py -- the central charge of the spinor/oscillator representation of vir
built on a CONJUGATE PAIR of vir-modules, in both statistics, and the answer to
The question:

   "Feigin interpreted the value of the central charge ... as the dimension of
    the EVEN PART of the superspace in which the superstring propagates. ...
    But what is the dimension of the odd part of the ambient space?"

The set-up is the one of the text (Subsection ssHatL) with the single self-dual
module W replaced by the pair

    W(lambda; mu) = cF_{lambda; mu}  (+)  cF_{1-lambda; -mu},

which is what one copy of the ambient superspace contributes: the residue
pairing pairs the two summands, and NOTHING inside a summand.  Explicitly, in
the bases  phi_i = t^{i+mu} (dt)^lambda  and  psi_k = t^{k-mu} (dt)^{1-lambda},

    L_j phi_i = (  mu + i + lambda      (j+1)) phi_{i+j},
    L_j psi_k = (- mu + k + (1-lambda)  (j+1)) psi_{k+j},
    B(phi_i, psi_k) = delta_{i+k, -1},   B(psi_k, phi_i) = eta delta_{i+k, -1},

with eta = +1 (a symmetric form: the generators are ODD, Clifford algebra,
spinor representation Spin) or eta = -1 (an anti-symmetric form: the generators
are EVEN, Weyl algebra, oscillator representation Osc).  Both are available for
one and the same pair of vir-modules -- that is the whole point: the two
systems differ ONLY by the parity of the coefficient superspace.

Invariance of B is verified below, as is  [hat L_j, e] = L_j e  and the
Virasoro relations; then (c, h) is read off the vacuum exactly as in fock.py:

    h = -(1/2) <L_1 L_{-1} One>,     c = 2 (<L_a L_{-a} One> + 2a h) * 12/(a^3-a)

for a = 2, 3, 4 independently.

Everything is exact arithmetic over Q.  Nothing is quoted from the literature.

What comes out (see run-target.log):

    c_Spin(lambda) = -2 (6 lambda^2 - 6 lambda + 1) = -c_Osc(lambda),

independent of mu and of the choice of the vacuum.  Hence, for an ambient
superspace V of sdim (d_ev | d_od) carrying an even non-degenerate form,

    c(cF (x) V) = d_ev c_1 + d_od (-c_1) = (d_ev - d_od) c_1 = sdim V * c_1 :

the central charge sees the SUPERdimension of V and nothing else.
"""

from fractions import Fraction as F
from itertools import product

W = 40                                   # index window; stability is checked


# ------------------------------------------------------------------ the case

class Pair:
    """W(lambda; mu) = cF_{lambda;mu} (+) cF_{1-lambda;-mu}, one statistics."""

    def __init__(self, lam, mu, form):
        self.lam, self.mu = F(lam), F(mu)
        self.form = form                 # 'sym' -> Clifford/Spin, 'asym' -> Weyl/Osc
        self.parity = 'odd' if form == 'sym' else 'even'
        self.eta = F(1) if form == 'sym' else F(-1)

    def name(self):
        return ('lambda=%-6s mu=%-6s %s' %
                (self.lam, self.mu, 'Spin (Clifford)' if self.form == 'sym'
                 else 'Osc  (Weyl)    '))

    # ---- the vir-action:  L_j e_I = kappa(I, j) e_{I+j}
    def kappa(self, I, j):
        a, i = I
        if a == 0:
            return self.mu + i + self.lam * (j + 1)
        return -self.mu + i + (1 - self.lam) * (j + 1)

    def shift(self, I, j):
        return (I[0], I[1] + j)

    # ---- the pairing
    def B(self, I, J):
        (a, i), (b, k) = I, J
        if a == b or i + k != -1:
            return F(0)
        return F(1) if a == 0 else self.eta

    def sigma(self, I):
        a, i = I
        return (1 - a, -1 - i)

    def beta(self, I):
        """B(e_{sigma(I)}, e_I)."""
        return self.B(self.sigma(I), I)

    def bnorm(self, I):
        """B(e_I, e_{sigma(I)}): the coefficient with which the annihilator
        e_I acts as d/d e_{sigma(I)}."""
        return self.B(I, self.sigma(I))

    # ---- the L_0-weight of a mode, and the polarization it dictates
    def weight(self, I):
        return self.kappa(I, 0)

    def kind(self, I):
        w = self.weight(I)
        assert w != 0, ('a zero mode at %s: choose a generic mu' % (I,))
        return 'C' if w < 0 else 'A'


# ------------------------------------------------------------------- states
# A state is a dict {mono: coeff}; mono a tuple of creation labels, strictly
# increasing in the odd (Clifford) case, sorted with repetitions in the even
# (Weyl) one.

ONE = {(): F(1)}


def sadd(a, b, s=F(1)):
    r = dict(a)
    for k, v in b.items():
        r[k] = r.get(k, F(0)) + s * v
        if r[k] == 0:
            del r[k]
    return r


def create(case, g, st):
    out = {}
    for mono, co in st.items():
        if case.parity == 'odd':
            if g in mono:
                continue
            k = sum(1 for x in mono if x < g)
            new = tuple(sorted(mono + (g,)))
            out[new] = out.get(new, F(0)) + (-co if k % 2 else co)
        else:
            new = tuple(sorted(mono + (g,)))
            out[new] = out.get(new, F(0)) + co
    return {k: v for k, v in out.items() if v != 0}


def annih(case, g, coef, st):
    """coef * d/d e_g."""
    out = {}
    for mono, co in st.items():
        if case.parity == 'odd':
            if g not in mono:
                continue
            k = mono.index(g)
            new = mono[:k] + mono[k + 1:]
            out[new] = out.get(new, F(0)) + (-1 if k % 2 else 1) * coef * co
        else:
            m = mono.count(g)
            if m == 0:
                continue
            k = mono.index(g)
            new = mono[:k] + mono[k + 1:]
            out[new] = out.get(new, F(0)) + m * coef * co
    return {k: v for k, v in out.items() if v != 0}


def e(case, I, st):
    """the generator e_I acting on the Fock space."""
    if case.kind(I) == 'C':
        return create(case, I, st)
    return annih(case, case.sigma(I), case.bnorm(I), st)


def L(case, j, st, window=W):
    r"""hat L_j = (1/2) sum_I (kappa(I,j)/beta_I) e_{I+j} e_{sigma(I)}."""
    tot = {}
    for a in (0, 1):
        for i in range(-window, window + 1):
            I = (a, i)
            b = case.beta(I)
            if b == 0:
                continue
            ka = case.kappa(I, j)
            if ka == 0:
                continue
            part = e(case, case.shift(I, j), e(case, case.sigma(I), st))
            if part:
                tot = sadd(tot, part, F(1, 2) * ka / b)
    return tot


def coeff_of_one(st):
    return st.get((), F(0))


# ------------------------------------------------------- structural checks

def check_form(case, rng=4):
    """B is non-degenerate, of the stated symmetry, and vir-invariant."""
    bad = 0
    for a, b in product((0, 1), repeat=2):
        for i, k in product(range(-rng, rng), repeat=2):
            I, J = (a, i), (b, k)
            # symmetry
            s = case.eta if case.form == 'sym' else -case.eta
            # B(J, I) = +/- B(I, J):  sym -> +, asym -> -
            want = case.B(I, J) * (1 if case.form == 'sym' else -1)
            if case.B(J, I) != want:
                bad += 1
            # invariance:  B(L_j I, J) + B(I, L_j J) = 0
            for j in (-2, -1, 1, 2):
                v = (case.kappa(I, j) * case.B(case.shift(I, j), J) +
                     case.kappa(J, j) * case.B(I, case.shift(J, j)))
                if v != 0:
                    bad += 1
    del s
    return bad


def basis_states(case, nmax=2):
    creators = [I for I in [(a, i) for a in (0, 1) for i in range(-6, 6)]
                if case.kind(I) == 'C']
    creators = sorted(creators, key=lambda I: -case.weight(I))[:4]
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
    return uniq[:12]


def check_action(case):
    """[hat L_j, e_I] = kappa(I,j) e_{I+j} on the Fock space."""
    bad = 0
    for j in (-2, -1, 1, 2):
        for a in (0, 1):
            for i in range(-3, 3):
                I = (a, i)
                for st in basis_states(case, 2):
                    lhs = sadd(L(case, j, e(case, I, st)),
                               e(case, I, L(case, j, st)), F(-1))
                    rhs = e(case, case.shift(I, j), st)
                    if sadd(lhs, rhs, -case.kappa(I, j)):
                        bad += 1
    return bad


def check_virasoro(case):
    """[L_a, L_b] = (b-a) L_{a+b} for a+b != 0."""
    bad = 0
    for a, b in ((1, 2), (1, 3), (-1, 2), (-2, 3), (-1, -2), (2, 3), (1, -3)):
        for st in basis_states(case, 2):
            lhs = sadd(L(case, a, L(case, b, st)), L(case, b, L(case, a, st)), F(-1))
            if sadd(lhs, L(case, a + b, st), -(b - a)):
                bad += 1
    return bad


def ch(case):
    """(c, h) with the three independent readings of c."""
    for j in (1, 2, 3, 4):
        assert not L(case, j, ONE), (case.name(), 'L_%d One != 0' % j)
    h = -F(1, 2) * coeff_of_one(L(case, 1, L(case, -1, ONE)))
    cs = []
    for a in (2, 3, 4):
        val = coeff_of_one(L(case, a, L(case, -a, ONE)))
        cs.append((val + 2 * a * h) * 12 / F(a ** 3 - a))
    assert cs[0] == cs[1] == cs[2], (case.name(), [str(x) for x in cs])
    return cs[0], h


def stability(case):
    return [coeff_of_one(L(case, 2, L(case, -2, ONE, window=w), window=w))
            for w in (12, 24, W)]


def predicted(lam, form):
    v = 2 * (6 * F(lam) ** 2 - 6 * F(lam) + 1)
    return -v if form == 'sym' else v


# ----------------------------------------------------------------- the runs

LAMS = [F(0), F(1, 3), F(1, 2), F(1), F(3, 2), F(2), F(-1), F(5, 2)]
MUS = [F(1, 3), F(1, 5), F(2, 7)]


def main():
    print('=' * 78)
    print('target.py -- the central charge of a conjugate pair of vir-modules')
    print('=' * 78)
    print()
    print('structural checks (form, action, Virasoro), on a few cases:')
    for lam in (F(0), F(1, 2), F(2)):
        for form in ('sym', 'asym'):
            c = Pair(lam, F(1, 3), form)
            print('   %s : B %2d,  [L,e] %2d,  vir %2d,  window %s'
                  % (c.name(), check_form(c), check_action(c),
                     check_virasoro(c), [str(x) for x in stability(c)]))
    print()
    print('(c, h) for the pair W(lambda; mu) = cF_{lambda;mu} (+) cF_{1-lambda;-mu}')
    print('%-8s %-8s %-14s %-14s %-14s' %
          ('lambda', 'mu', 'c (Spin)', 'c (Osc)', 'h (Spin), h (Osc)'))
    bad = 0
    for lam in LAMS:
        for mu in MUS:
            cs, hs = ch(Pair(lam, mu, 'sym'))
            co, ho = ch(Pair(lam, mu, 'asym'))
            ok = (cs == predicted(lam, 'sym') and co == predicted(lam, 'asym')
                  and cs == -co)
            bad += 0 if ok else 1
            print('%-8s %-8s %-14s %-14s %s, %s   %s'
                  % (lam, mu, cs, co, hs, ho, '' if ok else '  <-- MISMATCH'))
    print()
    print('all values match  c_Spin = -2(6L^2-6L+1) = -c_Osc :  %d mismatches'
          % bad)
    print()
    print('the two self-dual modules of the text are the halves of these pairs:')
    print('   Spin(sqrt(Vol)) : c = (1/2) c_Spin(1/2) = %s   (text: 1/2)'
          % (predicted(F(1, 2), 'sym') / 2))
    print('   Osc(cF_0/C.1)   : c = (1/2) c_Osc (0)   = %s   (text: 1)'
          % (predicted(F(0), 'asym') / 2))
    print()
    print('the ambient superspace V, sdim V = (d_ev | d_od), even form on V:')
    print('   an even direction of V gives one self-dual system,  c = +c_1')
    print('   two odd directions give one conjugate pair of the OTHER')
    print('   statistics,                                        c = -2 c_1')
    print('   ==> c(cF (x) V) = (d_ev - d_od) c_1 = sdim(V) c_1')
    print()
    for c1, name in ((F(1), 'even ambient direction over witt (cF_0/C.1)'),
                     (F(1, 2), 'even ambient direction over k^L(1|1), psi-part')):
        print('   %-46s c_1 = %s' % (name, c1))
    print()
    print('so the critical value 26 (resp. 10, 2, 0) says  d_ev - d_od = 26')
    print('(resp. 10, 2, 0), NOT d_ev = 26: the odd dimension is invisible to')
    print('the central charge.')


if __name__ == '__main__':
    main()
