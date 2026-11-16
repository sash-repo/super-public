#!/usr/bin/env python3
r"""
spin.py -- the spinor representation Spin(W), W = cF_{lambda;mu} (+) cF*_{lambda;mu},
of k^L(1|n), and the value of its central charge(s).

The question: "Please compute spin module (table 4.4.3)
over k^'L(1|4) for cocycles with non-zero beta and gamma (and doublecheck for
non-zero alpha)."  The text claims, above the table, that for n = 4 the three
central charges vanish: c_1 = c_2 = c_3 = 0 for any Spin(cF_{lambda;mu}).

Everything below is exact arithmetic over Q (module of Fractions), no floats and
no external packages.  Nothing is taken on faith: the contact bracket, the field
K_f, the module structure, the residue pairing, the Clifford-Weyl relations and
the operator \hat X are each verified before they are used.

------------------------------------------------------------------ the set-up

Functions:            R^L(n) = C[t,t^{-1}] (x) Lambda(theta_1, ..., theta_n),
                      a dict {(a, S): coeff}, a in Z, S a sorted tuple.

Contact field (46):  K_f = (2-E)(f) d_t
                            - (-1)^{p(f)} sum_i (d_{th_i}f - th_i d_t f) d_{th_i}

Contact bracket (48):{f,g} = (2-E)(f) d_t g - d_t f (2-E)(g)
                            - (-1)^{p(f)} sum_i d_{th_i}f d_{th_i}g

Module equation 95:     cF_{lambda;mu} = t^mu R^L(n) alpha_1^{lambda/2}  (n > 0),
                                     = t^mu R^L(0) alpha_1^{lambda}    (n = 0),
   with  L_{K_f} alpha_1 = 2 d_t(f) alpha_1, so on u = g alpha_1^{lambda/2}

           K_f . u = ( K_f(g) + w * d_t(f) * g ) alpha_1^{lambda/2},
           w = lambda for n > 0 and w = 2 lambda for n = 0  -- this is eq. (27).

Duality (4.5):      Vol = cF_{2-n}, and the residue Res(g alpha^{(2-n)/2}) =
   the coefficient of t^{-1} theta_1...theta_n in g, pairs
           cF_{lambda;mu}  with  cF_{2-n-lambda;-mu}.

Spin:                 W = V (+) V*, B the supersymmetric form; Cl(Pi W) with
   [psi_m, psi_n] = B(m,n), the bracket being the ANTI-commutator on W_ev (those
   modes are fermionic after the parity shift) and the commutator on W_od (they
   are bosonic).  Polarisation: creation = the modes with t-index <= -1.

           \hat X = (1/2) sum_m (1/beta_m) psi_{X phi_m} psi_{sigma(m)},
           beta_m = B(phi_{sigma(m)}, phi_m),

and c(X,Y) = the scalar [\hat X, \hat Y] - \hat{[X,Y]}, read off the vacuum.
"""

from fractions import Fraction as Q
from itertools import combinations, product
import sys


# ==================================================================== algebra
# a "function" is a dict {(a, S): Q}; a is the power of t, S a sorted tuple of
# odd indices in 1..n.  The monomial is t^a theta_{S[0]} ... theta_{S[-1]}.

def clean(f):
    return {k: v for k, v in f.items() if v != 0}


def add(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, Q(0)) + v
    return clean(out)


def smul(c, f):
    c = Q(c)
    return clean({k: c * v for k, v in f.items()})


def merge_sign(S, T):
    """sign of theta^S theta^T -> theta^(S u T); None if S and T overlap."""
    if set(S) & set(T):
        return None, None
    sign, out = 1, list(S) + list(T)
    # bubble sort, counting transpositions
    for i in range(len(out)):
        for j in range(len(out) - 1 - i):
            if out[j] > out[j + 1]:
                out[j], out[j + 1] = out[j + 1], out[j]
                sign = -sign
    return sign, tuple(out)


def mul(f, g):
    out = {}
    for (a, S), u in f.items():
        for (b, T), v in g.items():
            s, ST = merge_sign(S, T)
            if s is None:
                continue
            out[(a + b, ST)] = out.get((a + b, ST), Q(0)) + s * u * v
    return clean(out)


def d_t(f, mu=Q(0)):
    """d/dt on t^{mu+a} theta^S."""
    return clean({(a - 1, S): (mu + a) * v for (a, S), v in f.items()})


def d_odd(i, f):
    """left derivative d/d theta_i."""
    out = {}
    for (a, S), v in f.items():
        if i not in S:
            continue
        k = S.index(i)
        out[(a, S[:k] + S[k + 1:])] = out.get((a, S[:k] + S[k + 1:]), Q(0)) \
            + (-1 if k % 2 else 1) * v
    return clean(out)


def parity(f):
    ps = {len(S) % 2 for (_, S) in f}
    if len(ps) > 1:
        raise ValueError('inhomogeneous parity: %s' % f)
    return ps.pop() if ps else 0


def nabla(f):
    """(2 - E)(f), E the odd Euler operator."""
    return clean({(a, S): (2 - len(S)) * v for (a, S), v in f.items()})


def bracket(f, g, n):
    res = add(mul(nabla(f), d_t(g)), smul(-1, mul(d_t(f), nabla(g))))
    s = {}
    for i in range(1, n + 1):
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return add(res, smul(-((-1) ** parity(f)), s))


def mono(a, S=(), c=1):
    return {(a, tuple(S)): Q(c)}


# ======================================================= the module cF_{l;mu}
# A module element is a dict {(a, S): Q} standing for t^{mu+a} theta^S times the
# fixed generator alpha_1^{lambda/2} (alpha_1^{lambda} when n = 0).

class Fmod:
    """cF_{lambda; mu} over k^L(1|n)."""

    def __init__(self, n, lam, mu):
        self.n, self.lam, self.mu = n, Q(lam), Q(mu)
        self.w = 2 * self.lam if n == 0 else self.lam    # eq. (27)

    def act(self, f, u):
        """K_f . u."""
        n, mu = self.n, self.mu
        out = mul(nabla(f), d_t(u, mu))
        s = {}
        for i in range(1, n + 1):
            coef = add(d_odd(i, f), smul(-1, mul(mono(0, (i,)), d_t(f))))
            s = add(s, mul(coef, d_odd(i, u)))
        out = add(out, smul(-((-1) ** parity(f)), s))
        return add(out, smul(self.w, mul(d_t(f), u)))

    def basis(self, lo, hi):
        return [(a, S) for a in range(lo, hi + 1)
                for r in range(self.n + 1) for S in combinations(range(1, self.n + 1), r)]


def Kf_on_function(f, g, n):
    """K_f(g) -- the vector field K_f applied to a function g (no density weight)."""
    out = mul(nabla(f), d_t(g))
    s = {}
    for i in range(1, n + 1):
        coef = add(d_odd(i, f), smul(-1, mul(mono(0, (i,)), d_t(f))))
        s = add(s, mul(coef, d_odd(i, g)))
    return add(out, smul(-((-1) ** parity(f)), s))


# ============================================================ the superspace W
# A mode is a triple (s, a, S): s = 0 for the summand V = cF_{lambda;mu},
# s = 1 for V* = cF_{2-n-lambda;-mu}.

class Wspace:
    def __init__(self, n, lam, mu):
        self.n = n
        self.V = Fmod(n, lam, mu)
        # the dual weight is fixed by  w(lambda) + w(lambda*) = 2 - n = the
        # multiplier of Vol, and w = 2 lambda for n = 0 but lambda for n > 0
        # -- the two lines of eq. (27) again.
        lam_d = (1 - Q(lam)) if n == 0 else (2 - n - Q(lam))
        self.Vd = Fmod(n, lam_d, -Q(mu))
        self.full = tuple(range(1, n + 1))

    # ---- parity of a mode (before the Clifford parity shift).
    # For n odd the residue pairs theta^S with theta^{S^c}, |S| + |S^c| = n,
    # i.e. it is an ODD form; the dual summand therefore carries the parity
    # shift Pi that eq. equation 96 mentions, without which B is not even.
    def par(self, m):
        return (len(m[2]) + (self.n if m[0] == 1 else 0)) % 2

    # ---- sigma: the unique partner of m under the residue pairing
    def sigma(self, m):
        s, a, S = m
        comp = tuple(i for i in self.full if i not in S)
        return (1 - s, -1 - a, comp)

    # ---- the residue pairing <v, w>, v in V, w in V*
    def res_pair(self, va, vS, wb, wT):
        if va + wb != -1:
            return Q(0)
        sgn, ST = merge_sign(vS, wT)
        if sgn is None or ST != self.full:
            return Q(0)
        return Q(sgn)

    # ---- the supersymmetric form B on W
    def B(self, m1, m2):
        (s1, a1, S1), (s2, a2, S2) = m1, m2
        if s1 == s2:
            return Q(0)
        if s1 == 0:
            return self.res_pair(a1, S1, a2, S2)
        # B(w, v) = (-1)^{p(v)p(w)} B(v, w)
        return (-1) ** (self.par(m1) * self.par(m2)) * self.res_pair(a2, S2, a1, S1)

    def beta(self, m):
        return self.B(self.sigma(m), m)

    # ---- the action of K_f on a mode, returned as {mode: coeff}
    def act(self, f, m):
        s, a, S = m
        u = {(a, S): Q(1)}
        r = (self.V if s == 0 else self.Vd).act(f, u)
        return {(s, k[0], k[1]): v for k, v in r.items()}

    def is_creation(self, m):
        return m[1] <= -1


# =================================================== the Clifford-Weyl algebra
# States: dicts {word: coeff}, word = a tuple of CREATION modes in canonical
# order (repetitions allowed for the bosonic ones).  psi_m has the shifted
# parity  tp(m) = par(m) + 1,  and
#         psi_m psi_n = (-1)^{tp(m) tp(n)} psi_n psi_m + B(m, n).

VAC = {(): Q(1)}


def tp(Wsp, m):
    return (Wsp.par(m) + 1) % 2


def sadd(a, b, s=Q(1)):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Q(0)) + s * v
        if out[k] == 0:
            del out[k]
    return out


def psi(Wsp, m, st):
    """left multiplication by psi_m."""
    if Wsp.is_creation(m):
        return _psi_create(Wsp, m, st)
    return _psi_annih(Wsp, m, st)


def _psi_create(Wsp, m, st):
    out = {}
    tm = tp(Wsp, m)
    for word, co in st.items():
        # insert m into the sorted word, collecting the Koszul sign
        pos, sgn = 0, 1
        while pos < len(word) and word[pos] < m:
            if tm and tp(Wsp, word[pos]):
                sgn = -sgn
            pos += 1
        if tm and pos < len(word) and word[pos] == m:
            continue                      # fermionic square: psi_m^2 = B(m,m)/2 = 0
        new = word[:pos] + (m,) + word[pos:]
        out[new] = out.get(new, Q(0)) + sgn * co
    return clean(out)


def _psi_annih(Wsp, m, st):
    """move psi_m (an annihilation mode) to the right; psi_m |0> = 0."""
    out = {}
    tm = tp(Wsp, m)
    for word, co in st.items():
        sgn = 1
        for k in range(len(word)):
            nk = word[k]
            b = Wsp.B(m, nk)
            if b:
                rest = word[:k] + word[k + 1:]
                out[rest] = out.get(rest, Q(0)) + sgn * b * co
            if tm and tp(Wsp, nk):
                sgn = -sgn
    return clean(out)


def hatX(Wsp, f, st, window):
    r"""\hat K_f applied to the state st."""
    tot = {}
    n = Wsp.n
    for s in (0, 1):
        for a in range(-window, window + 1):
            for r in range(n + 1):
                for S in combinations(range(1, n + 1), r):
                    m = (s, a, S)
                    b = Wsp.beta(m)
                    if b == 0:
                        continue
                    img = Wsp.act(f, m)
                    if not img:
                        continue
                    sm = Wsp.sigma(m)
                    part = psi(Wsp, sm, st)
                    if not part:
                        continue
                    for p_, co in img.items():
                        q = psi(Wsp, p_, part)
                        if q:
                            tot = sadd(tot, q, Q(1, 2) * co / b)
    return tot


# ======================================================================= tests

def test_algebra(n):
    """[K_f, K_g] = K_{f,g} as operators on functions."""
    keys = [(a, S) for a in (-2, -1, 0, 1, 2)
            for r in range(n + 1) for S in combinations(range(1, n + 1), r)]
    small = keys[:len(keys)] if n <= 2 else [k for k in keys if k[0] in (-1, 0, 1)]
    bad = 0
    for kf, kg in product(small, repeat=2):
        f, g = mono(*kf), mono(*kg)
        for kh in small:
            h = mono(*kh)
            lhs = add(Kf_on_function(f, Kf_on_function(g, h, n), n),
                      smul(-((-1) ** (parity(f) * parity(g))),
                           Kf_on_function(g, Kf_on_function(f, h, n), n)))
            rhs = Kf_on_function(bracket(f, g, n), h, n)
            if add(lhs, smul(-1, rhs)):
                bad += 1
    return bad


def test_module(Wsp, window=2):
    """[K_f, K_g] . u = K_{f,g} . u on both summands of W."""
    n = Wsp.n
    keys = [(a, S) for a in (-1, 0, 1)
            for r in range(n + 1) for S in combinations(range(1, n + 1), r)]
    bad = 0
    for mod in (Wsp.V, Wsp.Vd):
        for kf, kg in product(keys, repeat=2):
            f, g = mono(*kf), mono(*kg)
            for ku in mod.basis(-1, 1):
                u = {ku: Q(1)}
                lhs = add(mod.act(f, mod.act(g, u)),
                          smul(-((-1) ** (parity(f) * parity(g))),
                               mod.act(g, mod.act(f, u))))
                rhs = mod.act(bracket(f, g, n), u)
                if add(lhs, smul(-1, rhs)):
                    bad += 1
    return bad


def test_form(Wsp, window=3):
    """B is supersymmetric, even, and k^L(1|n)-invariant."""
    n = Wsp.n
    modes = [(s, a, S) for s in (0, 1) for a in range(-window, window + 1)
             for r in range(n + 1) for S in combinations(range(1, n + 1), r)]
    bad_sym = bad_inv = 0
    for m1, m2 in product(modes, repeat=2):
        b12, b21 = Wsp.B(m1, m2), Wsp.B(m2, m1)
        if b12 != (-1) ** (Wsp.par(m1) * Wsp.par(m2)) * b21:
            bad_sym += 1
        if b12 and Wsp.par(m1) != Wsp.par(m2):
            bad_sym += 1
    keys = [(a, S) for a in (-1, 0, 1)
            for r in range(n + 1) for S in combinations(range(1, n + 1), r)]
    inner = [m for m in modes if abs(m[1]) <= window - 2]
    for kf in keys:
        f = mono(*kf)
        pf = parity(f)
        for m1, m2 in product(inner, repeat=2):
            tot = Q(0)
            for p_, co in Wsp.act(f, m1).items():
                tot += co * Wsp.B(p_, m2)
            sgn = (-1) ** (pf * Wsp.par(m1))
            for p_, co in Wsp.act(f, m2).items():
                tot += sgn * co * Wsp.B(m1, p_)
            if tot != 0:
                bad_inv += 1
    return bad_sym, bad_inv


def test_hat(Wsp, window, fs):
    r"""[\hat X, psi_v] = psi_{Xv} for even X."""
    n = Wsp.n
    modes = [(s, a, S) for s in (0, 1) for a in (-2, -1, 0, 1)
             for r in range(n + 1) for S in combinations(range(1, n + 1), r)]
    bad = 0
    for f in fs:
        if parity(f):
            continue
        for m in modes:
            lhs = sadd(hatX(Wsp, f, psi(Wsp, m, VAC), window),
                       psi(Wsp, m, hatX(Wsp, f, VAC, window)), Q(-1))
            rhs = {}
            for p_, co in Wsp.act(f, m).items():
                rhs = sadd(rhs, psi(Wsp, p_, VAC), co)
            if sadd(lhs, rhs, Q(-1)):
                bad += 1
    return bad


# ================================================================== the cocycle

def cocycle(Wsp, f, g, window):
    r"""the scalar [\hat K_f, \hat K_g] - \hat K_{f,g}, read on the vacuum.

    Only correct as it stands when both f and g are even, which is all we need.
    """
    lhs = sadd(hatX(Wsp, f, hatX(Wsp, g, VAC, window), window),
               hatX(Wsp, g, hatX(Wsp, f, VAC, window), window), Q(-1))
    br = bracket(f, g, Wsp.n)
    if br:
        lhs = sadd(lhs, hatX(Wsp, br, VAC, window), Q(-1))
    rest = {k: v for k, v in lhs.items() if k != ()}
    return lhs.get((), Q(0)), rest


def vacuum_weight(Wsp, window):
    """h = the value of ell_0 = (1/2) K_t on the vacuum."""
    v = hatX(Wsp, mono(1), VAC, window)
    rest = {k: c for k, c in v.items() if k != ()}
    return Q(1, 2) * v.get((), Q(0)), rest


# ------------------------------------------- the odd case (E^i, F^i entries)

def scommutator(Wsp, f, g, window):
    r"""[\hat K_f, \hat K_g] - \hat K_{f,g} on the vacuum, f, g of any parity.

    For odd f and g the supercommutator is the ANTI-commutator.
    """
    pf, pg = parity(f), parity(g)
    s = -((-1) ** (pf * pg))          # +1 when both are odd
    lhs = sadd(hatX(Wsp, f, hatX(Wsp, g, VAC, window), window),
               hatX(Wsp, g, hatX(Wsp, f, VAC, window), window), Q(s))
    br = bracket(f, g, Wsp.n)
    if br:
        lhs = sadd(lhs, hatX(Wsp, br, VAC, window), Q(-1))
    return lhs.get((), Q(0)), {k: v for k, v in lhs.items() if k != ()}


def test_hat_odd(Wsp, window, fs):
    r"""[\hat X, psi_v] = psi_{Xv} in the super sense, for X of any parity."""
    n = Wsp.n
    modes = [(s, a, S) for s in (0, 1) for a in (-2, -1, 0, 1)
             for r in range(n + 1) for S in combinations(range(1, n + 1), r)]
    bad = 0
    for f in fs:
        pf = parity(f)
        for m in modes:
            sgn = -((-1) ** (pf * tp(Wsp, m)))
            lhs = sadd(hatX(Wsp, f, psi(Wsp, m, VAC), window),
                       psi(Wsp, m, hatX(Wsp, f, VAC, window)), Q(sgn))
            rhs = {}
            for p_, co in Wsp.act(f, m).items():
                rhs = sadd(rhs, psi(Wsp, p_, VAC), co)
            if sadd(lhs, rhs, Q(-1)):
                bad += 1
    return bad
