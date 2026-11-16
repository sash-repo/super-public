#!/usr/bin/env python3
r"""
ns1.py -- the analog of Proposition 7.3 for the two stringy superalgebras
with ONE odd indeterminate,

        ns = k^L(1|1)   (Neveu-Schwarz, untwisted),
        r  = k^M(1|1)   (Ramond, twisted),

and the spinor representation Spin(W) for the doubled module

        W = cF_{lambda;mu} (+) (cF_{lambda;mu})^#.

This answers the question:
  "try to consider the analog of Proposition 7.3 for ns and r.  For 1 odd
   indeterminate, to begin with.  Next, consider Spin(W), where
   W = cF_{lambda;mu} (+) (cF_{lambda;mu})^# (also for 1 odd indeterminate)".

Everything is exact rational arithmetic.  Nothing is assumed: the bracket of
each of the two algebras, the module axiom for cF_{lambda;mu}, the invariance
of every form found, and the Clifford-Weyl relations are verified before use.

--------------------------------------------------------------------- set-up

Functions: R(1) = C[t,t^{-1}] (x) Lambda(w), w = theta (NS) or zeta (43); a dict
{(a, S): Q} with S = () or (1,), standing for t^a w^S.

NS, k^L(1|1)  (eq. (46), (48) of the text):
    K_f = nabla(f) d_t - (-1)^{p(f)} (d_w f - w d_t f) d_w,   nabla = 2 - E,
    {f,g} = nabla(f) d_t g - d_t f nabla(g) - (-1)^{p(f)} d_w f d_w g.

R, k^M(1|1)  (eq. (K^M), (K^Ma), (58), equation 60, equation 61):
    D = d_t - (w/2t) d_w,   E = w d_w,   nabla = 2 - E,
    K^M_f = nabla(f) D + (-1)^{p(f)} (1/t) (d_w f) d_w + D(f) E,
    {f,g}_R = nabla(f) D(g) - D(f) nabla(g) + s (-1)^{p(f)} kz (1/t) d_w f d_w g,
with (s, kz) fixed below by the Jacobi identity and by [K_f, K_g] = K_{{f,g}}.

Modules.  cF_{lambda;mu} = t^mu R(1) alpha^{lambda/2}, alpha = alpha_1 (NS) or
the Moebius form (43); the multiplier of alpha under K_f is 2 d_t(f), resp.
2 D(f), so
        K_f . u = (K_f(u) + lambda d_t(f) u)      (NS),
        K^M_f . u = (K^M_f(u) + lambda D(f) u)    (43).

The parameter mu matters modulo 1 in the NS sector and modulo 1 in the R sector
as well (t^mu R(1) as a module over R(1) with integer powers of t).
"""

from fractions import Fraction as Q
from itertools import product
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))
from exactlin import nullspace, rank                      # noqa: E402

ODD = 1                     # the single odd index


# ============================================================ the algebra R(1)

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


def mul(f, g):
    """product in C[t,t^{-1}] (x) Lambda(w)."""
    out = {}
    for (a, S), u in f.items():
        for (b, T), v in g.items():
            if S and T:
                continue                        # w^2 = 0
            ST = S + T
            out[(a + b, ST)] = out.get((a + b, ST), Q(0)) + u * v
    return clean(out)


def mono(a, S=(), c=1):
    return {(a, tuple(S)): Q(c)}


def d_t(f, mu=Q(0)):
    return clean({(a - 1, S): (mu + a) * v for (a, S), v in f.items()})


def d_w(f):
    """left derivative d/dw."""
    return clean({(a, ()): v for (a, S), v in f.items() if S})


def parity(f):
    ps = {len(S) % 2 for (_, S) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous parity: %s" % f)
    return ps.pop() if ps else 0


def E(f):
    return clean({(a, S): v for (a, S), v in f.items() if S})


def nabla(f):
    return clean({(a, S): (2 - len(S)) * v for (a, S), v in f.items()})


def D_R(f, mu=Q(0)):
    """D = d_t - (w/2t) d_w  in the Ramond sector."""
    return add(d_t(f, mu), smul(Q(-1, 2), mul(mono(-1, (ODD,)), d_w(f))))


# --------------------------------------------------------------- the brackets

def br_NS(f, g):
    res = add(mul(nabla(f), d_t(g)), smul(-1, mul(d_t(f), nabla(g))))
    return add(res, smul(-((-1) ** parity(f)), mul(d_w(f), d_w(g))))


def br_R(f, g, s=1, kz=Q(1)):
    res = add(mul(nabla(f), D_R(g)), smul(-1, mul(D_R(f), nabla(g))))
    odd = smul(kz, mul(mono(-1), mul(d_w(f), d_w(g))))
    return add(res, smul(s * ((-1) ** parity(f)), odd))


# ------------------------------------------------- the action on cF_{lambda;mu}

def act_NS(lam, mu, f, u):
    """K_f . u  for u in cF_{lambda;mu} over k^L(1|1)."""
    out = mul(nabla(f), d_t(u, mu))
    coef = add(d_w(f), smul(-1, mul(mono(0, (ODD,)), d_t(f))))
    out = add(out, smul(-((-1) ** parity(f)), mul(coef, d_w(u))))
    return add(out, smul(lam, mul(d_t(f), u)))


def act_R(lam, mu, f, u, s=1, kz=Q(1)):
    """K^M_f . u  for u in cF_{lambda;mu} over k^M(1|1)."""
    out = mul(nabla(f), D_R(u, mu))
    odd = smul(kz, mul(mono(-1), mul(d_w(f), d_w(u))))
    out = add(out, smul(s * ((-1) ** parity(f)), odd))
    out = add(out, mul(D_R(f), E(u)))
    return add(out, smul(lam, mul(D_R(f), u)))


# =============================================================== verification

def keys(arange, odd=(0, 1)):
    return [(a, () if r == 0 else (ODD,)) for a in arange for r in odd]


def jacobi_bad(bracket, ks):
    bad = 0
    for kf, kg, kh in product(ks, repeat=3):
        f, g, h = mono(*kf), mono(*kg), mono(*kh)
        lhs = bracket(f, bracket(g, h))
        rhs = add(bracket(bracket(f, g), h),
                  smul((-1) ** (parity(f) * parity(g)), bracket(g, bracket(f, h))))
        if add(lhs, smul(-1, rhs)):
            bad += 1
    return bad


def module_bad(act, bracket, lam, mu, ks, us):
    """[K_f, K_g].u = K_{f,g}.u  -- the module axiom."""
    bad = 0
    for kf, kg in product(ks, repeat=2):
        f, g = mono(*kf), mono(*kg)
        for ku in us:
            u = {ku: Q(1)}
            lhs = add(act(lam, mu, f, act(lam, mu, g, u)),
                      smul(-((-1) ** (parity(f) * parity(g))),
                           act(lam, mu, g, act(lam, mu, f, u))))
            rhs = act(lam, mu, bracket(f, g), u)
            if add(lhs, smul(-1, rhs)):
                bad += 1
    return bad


# ===================================================== invariant bilinear forms
#
# Unknowns: B(u_i, v_j), i, j running over a window of the basis of cF_{l;m},
# resp. cF_{l';m'}.  Equation, for every generator X = K_f and every pair:
#
#       B(X u, v) + (-1)^{p(X) p(u)} B(u, X v) = 0.
#
# An equation is used only if every unknown it mentions is inside the window
# (so no boundary term is silently set to 0); stability is checked by running
# two window sizes.

def weights(act, lam, mu, bas):
    """The eigenvalue of K_t on each basis element (K_t acts diagonally; this
    is verified, not assumed)."""
    t = mono(1)
    out = {}
    for ku in bas:
        img = act(lam, mu, t, {ku: Q(1)})
        if set(img) - {ku}:
            raise ValueError("K_t is not diagonal on %s" % (ku,))
        out[ku] = img.get(ku, Q(0))
    return out


def invariant_forms(act, lam, mu, lam2, mu2, N=5, gens=None, bracket=None):
    """Basis of the space of invariant bilinear forms cF_{l;m} x cF_{l2;m2} -> C.

    Returns (basis, index) where basis is a list of dicts {(ku, kv): Q} and
    index maps (ku, kv) -> column number.

    Invariance under K_t alone (which acts diagonally) kills every pair whose
    two eigenvalues do not add up to 0; only those pairs are carried as
    unknowns, which is what makes the system small.
    """
    bas = keys(range(-N, N + 1))
    w1 = weights(act, lam, mu, bas)
    w2 = weights(act, lam2, mu2, bas)
    idx = {}
    for ku in bas:
        for kv in bas:
            if w1[ku] + w2[kv] == 0:
                idx[(ku, kv)] = len(idx)
    if not idx:
        return [], idx
    if gens is None:
        gens = keys(range(-2, 3))
    inwin = set(bas)
    rows = []
    for kf in gens:
        f = mono(*kf)
        pf = parity(f)
        for ku in bas:
            for kv in bas:
                row = [Q(0)] * len(idx)
                ok = True
                for kp, co in act(lam, mu, f, {ku: Q(1)}).items():
                    if kp not in inwin:
                        ok = False           # boundary: the equation is unusable
                        break
                    if (kp, kv) in idx:
                        row[idx[(kp, kv)]] += co
                    # else: that unknown is 0 by the K_t-weight argument
                if not ok:
                    continue
                sgn = (-1) ** (pf * (len(ku[1]) % 2))
                for kp, co in act(lam2, mu2, f, {kv: Q(1)}).items():
                    if kp not in inwin:
                        ok = False
                        break
                    if (ku, kp) in idx:
                        row[idx[(ku, kp)]] += sgn * co
                if not ok:
                    continue
                if any(row):
                    rows.append(row)
    if not rows:
        rows = [[Q(0)] * len(idx)]
    ns = nullspace(rows)
    inv = {v: k for k, v in idx.items()}
    out = []
    for vec in ns:
        d = {}
        for j, c in enumerate(vec):
            if c:
                d[inv[j]] = Q(c)
        if d:
            out.append(d)
    return out, idx


def form_report(form, N):
    """(parity, symmetry, nondegenerate-on-the-inner-window) of a form."""
    pars = {(len(ku[1]) + len(kv[1])) % 2 for (ku, kv) in form}
    par = pars.pop() if len(pars) == 1 else None
    # symmetry: B(u,v) = eps (-1)^{p(u)p(v)} B(v,u)
    eps = set()
    for (ku, kv), c in form.items():
        c2 = form.get((kv, ku), Q(0))
        sg = (-1) ** ((len(ku[1]) % 2) * (len(kv[1]) % 2))
        if c2 == 0:
            eps.add("degenerate-pair")
        else:
            eps.add(Q(c) / (sg * c2))
    # Non-degeneracy: no basis vector of the inner window is annihilated by the
    # form.  The second slot runs over the WHOLE window, since the partner of an
    # inner vector need not be inner.
    inner = keys(range(-N + 3, N - 2))
    full = keys(range(-N, N + 1))
    dead = [ku for ku in inner
            if not any(form.get((ku, kv), Q(0)) for kv in full)]
    dead += [kv for kv in inner
             if not any(form.get((ku, kv), Q(0)) for ku in full)]
    return par, eps, (not dead), dead


# ======================================================= the doubled module W
#
# W = V (+) V*, V = cF_{lambda;mu}, V* = cF_{lam_d; -mu} with the invariant
# pairing found above.  Modes: (s, a, S), s = 0 for V, s = 1 for V*.

class Wspace:
    def __init__(self, sector, lam, mu, lam_d, pair, par_shift):
        self.sector = sector
        self.n = 1
        self.lam, self.mu, self.lam_d = Q(lam), Q(mu), Q(lam_d)
        self.pair = pair            # {(ku, kv): Q}, ku in V, kv in V*
        self.par_shift = par_shift  # parity shift of the V* summand
        self.act_ = act_NS if sector == "NS" else act_R
        self._sig = {}
        for (ku, kv), c in pair.items():
            if c:
                self._sig[(0,) + ku] = (1,) + kv
                self._sig[(1,) + kv] = (0,) + ku

    def par(self, m):
        return (len(m[2]) + (self.par_shift if m[0] == 1 else 0)) % 2

    def sigma(self, m):
        return self._sig[m]

    def B(self, m1, m2):
        (s1, a1, S1), (s2, a2, S2) = m1, m2
        if s1 == s2:
            return Q(0)
        if s1 == 0:
            return self.pair.get((((a1, S1)), ((a2, S2))), Q(0))
        return (-1) ** (self.par(m1) * self.par(m2)) * \
            self.pair.get((((a2, S2)), ((a1, S1))), Q(0))

    def beta(self, m):
        return self.B(self.sigma(m), m)

    def act(self, f, m):
        s, a, S = m
        u = {(a, S): Q(1)}
        lam = self.lam if s == 0 else self.lam_d
        mu = self.mu if s == 0 else -self.mu
        r = self.act_(lam, mu, f, u)
        return {(s, k[0], k[1]): v for k, v in r.items()}

    def is_creation(self, m):
        return m[1] <= -1

    def modes(self, window):
        return [(s, a, S) for s in (0, 1) for a in range(-window, window + 1)
                for S in ((), (ODD,))]


# ---------------------------------------------------- the Clifford-Weyl algebra
# psi_m has the SHIFTED parity tp(m) = par(m) + 1 and
#      psi_m psi_n = (-1)^{tp(m) tp(n)} psi_n psi_m + B(m, n).

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
    if Wsp.is_creation(m):
        return _psi_create(Wsp, m, st)
    return _psi_annih(Wsp, m, st)


def _psi_create(Wsp, m, st):
    out = {}
    tm = tp(Wsp, m)
    for word, co in st.items():
        pos, sgn = 0, 1
        while pos < len(word) and word[pos] < m:
            if tm and tp(Wsp, word[pos]):
                sgn = -sgn
            pos += 1
        if tm and pos < len(word) and word[pos] == m:
            continue                       # fermionic square
        new = word[:pos] + (m,) + word[pos:]
        out[new] = out.get(new, Q(0)) + sgn * co
    return {k: v for k, v in out.items() if v != 0}


def _psi_annih(Wsp, m, st):
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
    return {k: v for k, v in out.items() if v != 0}


def hatX(Wsp, f, st, window):
    tot = {}
    for m in Wsp.modes(window):
        b = Wsp.beta(m)
        if b == 0:
            continue
        img = Wsp.act(f, m)
        if not img:
            continue
        part = psi(Wsp, Wsp.sigma(m), st)
        if not part:
            continue
        for p_, co in img.items():
            q = psi(Wsp, p_, part)
            if q:
                tot = sadd(tot, q, Q(1, 2) * co / b)
    return tot


def test_hat(Wsp, window, fs):
    """[hat X, psi_v] = psi_{Xv} (super sense) -- the defining property."""
    bad = 0
    for f in fs:
        pf = parity(f)
        for m in Wsp.modes(2):
            sgn = -((-1) ** (pf * tp(Wsp, m)))
            lhs = sadd(hatX(Wsp, f, psi(Wsp, m, VAC), window),
                       psi(Wsp, m, hatX(Wsp, f, VAC, window)), Q(sgn))
            rhs = {}
            for p_, co in Wsp.act(f, m).items():
                rhs = sadd(rhs, psi(Wsp, p_, VAC), co)
            if sadd(lhs, rhs, Q(-1)):
                bad += 1
    return bad


def cocycle(Wsp, bracket, f, g, window):
    """the scalar [hat K_f, hat K_g] - hat K_{f,g} on the vacuum (f, g even)."""
    lhs = sadd(hatX(Wsp, f, hatX(Wsp, g, VAC, window), window),
               hatX(Wsp, g, hatX(Wsp, f, VAC, window), window), Q(-1))
    br = bracket(f, g)
    if br:
        lhs = sadd(lhs, hatX(Wsp, br, VAC, window), Q(-1))
    return lhs.get((), Q(0)), {k: v for k, v in lhs.items() if k != ()}
