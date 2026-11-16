#!/usr/bin/env python3
r"""
fock.py -- exact (rational) computation of the pair (c, h) for the four
spinor/oscillator representations of vir listed in the Lemma of the section
"Examples of spinor/oscillator representations" of the text, i.e. the Lemma that
carries two doubtful terms ("+8L_0", "-6").

The four self-dual vir-modules of Statement 4.5 (= Proposition 7.3):

  1) sqrt(Vol)        = F_{1/2}       symmetric pairing, ODD  elements (spinor)
  2) sqrt(t Vol)      = F_{1/2, 1/2}  symmetric pairing, ODD  elements (spinor)
  3) F_0 / C.1        = F_{0}/C       anti-symmetric,    EVEN elements (osc)
  4) sqrt(t) F_0      = F_{0, 1/2}    anti-symmetric,    EVEN elements (osc)

Set-up (all of it forced by the text, nothing chosen by hand):

  basis of F_{lambda, mu}:  phi_i = t^{i+mu} (dt)^lambda,  i in Z
  vir action:               L_j(phi_i) = kappa(i,j) phi_{i+j},
                            kappa(i,j) = mu + i + lambda (j+1)          (eq. 56)
  symmetric pairing:        B(phi_i, phi_j) = Res phi_i phi_j           equation 143
  anti-symmetric pairing:   B(phi_i, phi_j) = Res phi_i d(phi_j)        equation 144

The pairing gives the duality involution sigma (B(phi_i, phi_j) != 0 iff
j = sigma(i)) and the numbers beta_i := B(phi_{sigma(i)}, phi_i).  Creation
operators are the phi_i with sigma(i) > i, annihilation operators those with
sigma(i) < i; an index with sigma(i) = i (this happens exactly once, in case 2)
is *self-dual* -- that is the "theta" of the case o(2k+1), and the reason why
the naive rule "f_i^* = d/d f_i" of the text breaks down there.

The Clifford/Weyl algebra is normalised, as in the text, by

  {d/d c_i, c_j} = delta_ij   resp.  [d/d c_i, c_j] = delta_ij,

i.e. by  {w, w'} = B(w, w')  resp.  [w, w'] = B(w, w'); hence a self-dual w
has w^2 = B(w,w)/2, which is realised below by the parity operator.

The operator representing X in so(W) resp. sp(W) is

  hat X = (1/2) sum_alpha X(e_alpha) e^alpha,      B(e^alpha, e_beta) = delta,

the unique quadratic element with [hat X, w] = X(w) for all w in W (both
identities are verified below, on the nose).  For X = L_j this reads

  L_j = (1/2) sum_{i in Z} (kappa(i,j)/beta_i) phi_{i+j} phi_{sigma(i)} .   (*)

The factor 1/2 is NOT optional: in (*) every monomial phi_a phi_b occurs twice
(for i and for i' = sigma(i) - j).  Dropping it -- as the displayed formulas
   L_j = sum_i kappa(i,j) phi_{i+j} d/d phi_i
of the text do, since they let phi_{i+j} run over the whole of Z while reading
d/d phi_i as phi_{sigma(i)} -- yields 2 L_j, which does not satisfy the
Virasoro relations at all; this is verified below as well.

Given (*), everything is unambiguous:  L_j One = 0 for j > 0, and

  h = -(1/2) L_1 L_{-1} One,      c = 2 (L_2 L_{-2} One + 4 h),

the latter from [L_2, L_{-2}] = -4 L_0 + (1/2) z, see equation 148; the value of c
is cross-checked against [L_3, L_{-3}] = -6 L_0 + 2 z and [L_4, L_{-4}].
"""

from fractions import Fraction as F
from itertools import product

W = 40          # window for the (infinite) sums over i in (*)


# --------------------------------------------------------------- the 4 cases

class Case:
    def __init__(self, name, lam, mu, parity, form, drop=()):
        self.name, self.lam, self.mu = name, F(lam), F(mu)
        self.parity = parity                 # 'odd' (Clifford) / 'even' (Weyl)
        self.form = form                     # 'sym' / 'asym'
        self.drop = set(drop)                # indices killed in the quotient
        self._sig = {}

    # ---- the pairing B(phi_i, phi_j)
    def B(self, i, j):
        if i in self.drop or j in self.drop:
            return F(0)
        if self.form == 'sym':               # Res t^{i+j+2mu} (dt)^{2lambda}
            return F(1) if i + j + 2 * self.mu == -1 else F(0)
        else:                                # Res phi_i d(phi_j)
            return (j + self.mu) if i + j + 2 * self.mu == 0 else F(0)

    def sigma(self, i):
        if i not in self._sig:
            s = None
            if i not in self.drop:
                # B(i, j) != 0 forces i + j + 2mu = -1 (sym) resp. = 0 (asym)
                q = -i - 2 * self.mu - (1 if self.form == 'sym' else 0)
                if q == int(q) and self.B(i, int(q)) != 0:
                    s = int(q)
            self._sig[i] = s
        return self._sig[i]

    def beta(self, i):
        """B(phi_{sigma(i)}, phi_i): the normalisation of the B-dual basis."""
        s = self.sigma(i)
        return None if s is None else self.B(s, i)

    def bnorm(self, i):
        """B(phi_i, phi_{sigma(i)}): the (anti)commutator of an annihilation
        operator with its own creation operator, i.e. the coefficient with
        which phi_i acts as d/d phi_{sigma(i)}.  For an anti-symmetric B this
        is MINUS beta(i)."""
        s = self.sigma(i)
        return None if s is None else self.B(i, s)

    def kappa(self, i, j):
        return self.mu + i + self.lam * (j + 1)

    def kind(self, i):
        s = self.sigma(i)
        if s is None:
            return 'none'
        return 'C' if s > i else ('A' if s < i else 'S')


CASES = [
    Case('sqrt(Vol)      = F_{1/2}      (spinor)', F(1, 2), 0, 'odd', 'sym'),
    Case('sqrt(t Vol)    = F_{1/2,1/2}  (spinor)', F(1, 2), F(1, 2), 'odd', 'sym'),
    Case('F_0/C.1        = F_0/C        (osc)   ', 0, 0, 'even', 'asym', drop=(0,)),
    Case('sqrt(t) F_0    = F_{0,1/2}    (osc)   ', 0, F(1, 2), 'even', 'asym'),
]


# ------------------------------------------------------------------- states
#
# A state is a dict {(mono, th): coeff}; mono is a tuple of creation labels --
# strictly increasing in the odd case, sorted with repetitions in the even one
# -- and th in {0, 1} is the exponent of the self-dual generator theta, which
# is written to the RIGHT of mono.  theta acts as (-1)^{deg mono} times
# multiplication by theta, with theta^2 = beta_theta / 2: that is the parity
# (Gamma) realisation of an odd-dimensional Clifford module.

ONE = {((), 0): F(1)}


def sadd(a, b, s=F(1)):
    r = dict(a)
    for k, v in b.items():
        r[k] = r.get(k, F(0)) + s * v
        if r[k] == 0:
            del r[k]
    return r


def create(case, g, st):
    """multiplication by the generator phi_g (g a creation label)."""
    out = {}
    for (mono, th), co in st.items():
        if case.parity == 'odd':
            if g in mono:
                continue
            k = sum(1 for x in mono if x < g)      # move phi_g past k factors
            new = tuple(sorted(mono + (g,)))
            sgn = -1 if k % 2 else 1
            out[(new, th)] = out.get((new, th), F(0)) + sgn * co
        else:
            new = tuple(sorted(mono + (g,)))
            out[(new, th)] = out.get((new, th), F(0)) + co
    return {k: v for k, v in out.items() if v != 0}


def annih(case, g, coef, st):
    """coef * d/d phi_g  (the operator phi_i with sigma(i) = g < i)."""
    out = {}
    for (mono, th), co in st.items():
        if case.parity == 'odd':
            if g not in mono:
                continue
            k = mono.index(g)                       # d/d phi_g from the left
            new = mono[:k] + mono[k + 1:]
            sgn = -1 if k % 2 else 1
            out[(new, th)] = out.get((new, th), F(0)) + sgn * coef * co
        else:
            m = mono.count(g)
            if m == 0:
                continue
            k = mono.index(g)
            new = mono[:k] + mono[k + 1:]
            out[(new, th)] = out.get((new, th), F(0)) + m * coef * co
    return {k: v for k, v in out.items() if v != 0}


def selfdual(case, i, st):
    """the self-dual generator theta = phi_i, sigma(i) = i, theta^2 = beta/2."""
    b2 = case.B(i, i) / 2
    out = {}
    for (mono, th), co in st.items():
        sgn = -1 if (len(mono) % 2 and case.parity == 'odd') else 1
        if th == 0:
            key, val = (mono, 1), sgn * co
        else:
            key, val = (mono, 0), sgn * b2 * co
        out[key] = out.get(key, F(0)) + val
    return {k: v for k, v in out.items() if v != 0}


def phi(case, i, st):
    """the operator phi_i on the Fock space."""
    k = case.kind(i)
    if k == 'C':
        return create(case, i, st)
    if k == 'A':
        return annih(case, case.sigma(i), case.bnorm(i), st)
    if k == 'S':
        return selfdual(case, i, st)
    return {}


def L(case, j, st, half=True, window=W):
    """L_j of (*) applied to st; half=False gives the text's 2 L_j."""
    tot = {}
    for i in range(-window, window + 1):
        if i in case.drop or (i + j) in case.drop:
            continue
        b = case.beta(i)
        if b is None or b == 0:
            continue
        ka = case.kappa(i, j)
        if ka == 0:
            continue
        s = case.sigma(i)
        if s is None or (s in case.drop):
            continue
        part = phi(case, i + j, phi(case, s, st))
        if part:
            tot = sadd(tot, part, (F(1, 2) if half else F(1)) * ka / b)
    return tot


def coeff_of_one(st):
    return st.get(((), 0), F(0))


# ------------------------------------------------------- structural checks


def basis_states(case, nmax=3):
    """a few low states: products of the first creation generators."""
    creators = [i for i in range(-8, 8) if case.kind(i) == 'C'][-4:]
    out = [ONE]
    for r in range(1, nmax + 1):
        for combo in product(creators, repeat=r):
            st = ONE
            for g in combo:
                st = create(case, g, st)
            if st:
                out.append(st)
    # de-duplicate
    seen, uniq = set(), []
    for st in out:
        k = tuple(sorted((m, str(v)) for m, v in st.items()))
        if k not in seen:
            seen.add(k)
            uniq.append(st)
    return uniq[:14]


def check_action_on_W(case):
    """[hat L_j, phi_i] = kappa(i,j) phi_{i+j} on the Fock space."""
    bad = 0
    for j in (-2, -1, 1, 2):
        for i in range(-4, 4):
            if i in case.drop or case.beta(i) is None:
                continue
            for st in basis_states(case, 2):
                lhs = sadd(L(case, j, phi(case, i, st)),
                           phi(case, i, L(case, j, st)), F(-1))
                rhs = phi(case, i + j, st) if (i + j) not in case.drop else {}
                d = sadd(lhs, rhs, -case.kappa(i, j))
                if d:
                    bad += 1
    return bad


def check_virasoro(case, half=True):
    """[L_a, L_b] = (b-a) L_{a+b} for a+b != 0, on low states."""
    bad = 0
    for a, b in ((1, 2), (1, 3), (-1, 2), (-2, 3), (-1, -2), (2, 3), (1, -3)):
        for st in basis_states(case, 2):
            lhs = sadd(L(case, a, L(case, b, st, half), half),
                       L(case, b, L(case, a, st, half), half), F(-1))
            rhs = L(case, a + b, st, half)
            if sadd(lhs, rhs, -(b - a)):
                bad += 1
    return bad


def central_charge(case):
    """(c, h) plus the two independent cross-checks of c."""
    for j in (1, 2, 3, 4):
        assert not L(case, j, ONE), (case.name, 'L_%d One != 0' % j)
    h = -F(1, 2) * coeff_of_one(L(case, 1, L(case, -1, ONE)))
    cs = []
    for a in (2, 3, 4):
        val = coeff_of_one(L(case, a, L(case, -a, ONE)))
        # [L_a, L_{-a}] = -2a L_0 + (a^3-a)/12 z  on One:
        cs.append((val + 2 * a * h) * 12 / F(a ** 3 - a))
    return h, cs


def stability(case):
    """the window W must not matter."""
    out = []
    for w in (12, 24, W):
        v = coeff_of_one(L(case, 2, L(case, -2, ONE, window=w), window=w))
        out.append(v)
    return out


if __name__ == '__main__':
    print('=' * 78)
    print('(c, h) of the four self-dual vir-modules of Statement 4.5')
    print('=' * 78)
    for case in CASES:
        print()
        print(case.name)
        pol = {k: [i for i in range(-6, 6) if case.kind(i) == k]
               for k in 'CAS'}
        print('   creation phi_i: ... %s ;  annihilation: %s ... ;  self-dual: %s'
              % (pol['C'][-3:], pol['A'][:3], pol['S'] or '-'))
        print('   [L_j, phi_i] = kappa(i,j) phi_{i+j} :  %d failures'
              % check_action_on_W(case))
        print('   [L_a, L_b] = (b-a) L_{a+b}, a+b!=0  :  %d failures'
              % check_virasoro(case, half=True))
        print('   same for the text\'s 2 L_j            :  %d failures'
              % check_virasoro(case, half=False))
        print('   window-independence (W=12,24,40)     :  %s'
              % [str(x) for x in stability(case)])
        h, cs = central_charge(case)
        print('   h = %-6s   c = %s   (from [L_2,L_-2], [L_3,L_-3], [L_4,L_-4])'
              % (h, [str(x) for x in cs]))
        assert cs[0] == cs[1] == cs[2], 'c is not consistent!'
        print('   ==>  (c, h) = (%s, %s)        doubled: (%s, %s)'
              % (cs[0], h, 2 * cs[0], 2 * h))
