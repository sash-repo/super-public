#!/usr/bin/env python3
r"""
ns4.py -- FOUR odd indeterminates: the modules cF_{lambda; mu} that carry an
invariant bilinear form, and the highest weight of Spin and Osc built on them,
over

        k^L(1|4)   (Neveu-Schwarz, untwisted),
        k^M(1|4)   (Ramond,        twisted).

This answers the question:

  "In order not to deal with the odd pairing, let us consider the case of 4 odd
   indeterminates, for both NS and Ramond cases, when there is a bilinear form
   on cF_{lambda; mu}; compute the highest weight of Spin and Osci."

It is the sequel of ns1.py and ns2.py .
Everything is exact rational arithmetic; nothing is assumed: the bracket of
each algebra, the module axiom, the invariance of every form found, the
Clifford-Weyl relations and the defining property of the quadratic lift are all
verified before use.

--------------------------------------------------------------------- set-up

Functions.  R(4) = C[t, t^{-1}] (x) Lambda(w_1, ..., w_4); a monomial is a key
(a, S) with a in Q and S a sorted subtuple of (1, 2, 3, 4), standing for
t^a w^S.

  * k^L(1|4): the xi-eta realization with k = 2, see eq. (41) of the text:
        w_1 = xi_1, w_2 = eta_1, w_3 = xi_2, w_4 = eta_2,
    so that alpha_1 = dt + xi_1 d eta_1 + eta_1 d xi_1
                         + xi_2 d eta_2 + eta_2 d xi_2, see eq. (40).
    The maximal torus of the o(4) = <K_f | f quadratic in the odd
    indeterminates> is Span(K_{xi_1 eta_1}, K_{xi_2 eta_2}), see eq. (42),
    so the highest weight carries TWO labels H_1, H_2 besides (c, h).

  * k^M(1|4): n = 4 = 2k+2 with k = 1, see eq. equation 53, so the odd
    indeterminates are the twisted zeta, the left-over untwisted theta, and one
    xi-eta pair:
        w_1 = zeta, w_2 = theta, w_3 = xi, w_4 = eta,
    and tilde alpha = dt + t zeta d zeta + theta d theta + xi d eta + eta d xi.
    Here the torus is Span(K^M_{xi eta}), so there is ONE label H.

The brackets are eqs. (3a)/(4a) for k^L and eqs. (K^M), (10a), (12a) for k^M,
written for a general number of xi-eta pairs.

Modules.  cF_{lambda; mu} = t^mu cF alpha_1^{lambda/2}, see eq. equation 95, and
cF^M_{lambda; mu} = t^mu R^M(4) (tilde alpha)^{lambda/2}, see eq. equation 98.
The multiplier of alpha_1 is 2 d_t(f) and that of tilde alpha is 2 cD(f), so

        K_f . u   = K_f(u)   + lambda d_t(f) u,
        K^M_f . u = K^M_f(u) + lambda cD(f) u.

Unlike the case n = 2, there is NO second parameter here: k^L(1|4)_0 =
C K_t (+) o(4) with o(4) semisimple, so K_t is the only source of a character.
This is checked from scratch in runns4.py, cohomologically as well.

Duality.  Vol = (Pi)^n cF_{2-n} = cF_{-2} for n = 4 (no parity shift, n being
even), so the residue pairing matches cF_{lambda; mu} with cF_{-2-lambda; ?}:
the SELF-dual value is lambda = -1, and the pairing is EVEN, |S| + |S^c| = 4.
That is the point of the text's request: no odd pairing, hence a genuine Spin (and,
for the anti-supersymmetric form, a genuine Osc).

Spin versus Osc.  Given an even non-degenerate invariant form B on V, the
generators psi_v obey
        psi_m psi_n = (-1)^{tp(m) tp(n)} psi_n psi_m + B(m, n),
where tp(m) = p(m) + shift.  Consistency forces
        B(n, m) = (-1)^{p(m) p(n)} B(m, n)   for shift = 1  (Spin),
        B(n, m) = -(-1)^{p(m) p(n)} B(m, n)  for shift = 0  (Osc),
i.e., a supersymmetric B gives the spinor representation (Clifford on the even
vectors of V, Weyl on the odd ones) and an anti-supersymmetric B gives the
oscillator one.  Both are computed below.
"""

from fractions import Fraction as Q
from itertools import product, combinations
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))
from exactlin import nullspace, rank as exrank                # noqa: E402

NODD = 4
SUBSETS = tuple(tuple(S) for r in range(NODD + 1)
                for S in combinations(range(1, NODD + 1), r))
SRANK = {S: i for i, S in enumerate(SUBSETS)}

# --- k^L(1|4): w_1 = xi_1, w_2 = eta_1, w_3 = xi_2, w_4 = eta_2
LPAIRS = ((1, 2), (3, 4))
TORUS_L = ((Q(0), (1, 2)), (Q(0), (3, 4)))

# --- k^M(1|4): w_1 = zeta, w_2 = theta, w_3 = xi, w_4 = eta
MZETA, MTHETA = 1, 2
MPAIRS = ((3, 4),)
TORUS_M = ((Q(0), (3, 4)),)

NAMES_L = {1: "x1", 2: "y1", 3: "x2", 4: "y2"}
NAMES_M = {1: "z", 2: "h", 3: "x", 4: "y"}


# ============================================================ the algebra R(4)

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
    if c == 0:
        return {}
    return clean({k: c * v for k, v in f.items()})


def _sortsign(S):
    S = list(S)
    sign = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sign = -sign
    for j in range(len(S) - 1):
        if S[j] == S[j + 1]:
            return None, 0
    return tuple(S), sign


def mul(f, g):
    out = {}
    for (a, S), u in f.items():
        for (b, T), v in g.items():
            ST, sg = _sortsign(S + T)
            if ST is None:
                continue
            k = (a + b, ST)
            out[k] = out.get(k, Q(0)) + sg * u * v
    return clean(out)


def mono(a, S=(), c=1):
    return {(Q(a), tuple(S)): Q(c)}


def d_t(f, mu=Q(0)):
    return clean({(a - 1, S): (mu + a) * v for (a, S), v in f.items()})


def d_odd(f, i):
    """left derivative d/dw_i."""
    out = {}
    for (a, S), v in f.items():
        if i in S:
            k = S.index(i)
            T = S[:k] + S[k + 1:]
            out[(a, T)] = out.get((a, T), Q(0)) + ((-1) ** k) * v
    return clean(out)


def parity(f):
    ps = {len(S) % 2 for (_, S) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous parity: %s" % f)
    return ps.pop() if ps else 0


def cE(f):
    return clean({(a, S): len(S) * v for (a, S), v in f.items()})


def nabla(f):
    return clean({(a, S): (2 - len(S)) * v for (a, S), v in f.items()})


# ================================================================== k^L(1|4)

def br_L(f, g):
    res = add(mul(nabla(f), d_t(g)), smul(-1, mul(d_t(f), nabla(g))))
    odd = {}
    for (x, y) in LPAIRS:
        odd = add(odd, mul(d_odd(f, x), d_odd(g, y)),
                  mul(d_odd(f, y), d_odd(g, x)))
    return add(res, smul(-((-1) ** parity(f)), odd))


def K_L(f, u, mu=Q(0)):
    """the contact vector field K_f applied to a function u in t^mu R(4)."""
    p = parity(f)
    out = mul(nabla(f), d_t(u, mu))
    acc = {}
    for (x, y) in LPAIRS:
        cx = add(d_odd(f, y), smul(-1, mul(mono(0, (x,)), d_t(f))))
        cy = add(d_odd(f, x), smul(-1, mul(mono(0, (y,)), d_t(f))))
        acc = add(acc, mul(cx, d_odd(u, x)), mul(cy, d_odd(u, y)))
    return add(out, smul(-((-1) ** p), acc))


def act_L(par, f, u):
    """K_f . u on cF_{lambda; mu}; par = (lambda, mu)."""
    lam, mu = par
    out = K_L(f, u, mu)
    if lam:
        out = add(out, mul(smul(lam, d_t(f)), u))
    return out


# ================================================================== k^M(1|4)

def D_M(f, mu=Q(0)):
    """cD = d_t - (zeta / 2t) d_zeta."""
    return add(d_t(f, mu),
               smul(Q(-1, 2), mul(mono(-1, (MZETA,)), d_odd(f, MZETA))))


def _MP(f, g):
    """the Moebius-Poisson bracket, eq. (12a)."""
    mp = mul(mono(-1), mul(d_odd(f, MZETA), d_odd(g, MZETA)))
    mp = add(mp, mul(d_odd(f, MTHETA), d_odd(g, MTHETA)))
    for (x, y) in MPAIRS:
        mp = add(mp, mul(d_odd(f, x), d_odd(g, y)),
                 mul(d_odd(f, y), d_odd(g, x)))
    return mp


def br_M(f, g):
    res = add(mul(nabla(f), D_M(g)), smul(-1, mul(D_M(f), nabla(g))))
    return add(res, smul((-1) ** parity(f), _MP(f, g)))


def K_M(f, u, mu=Q(0)):
    p = parity(f)
    out = mul(nabla(f), D_M(u, mu))
    out = add(out, smul((-1) ** p, _MP(f, u)))
    return add(out, mul(D_M(f), cE(u)))


def act_M(par, f, u):
    """K^M_f . u on cF^M_{lambda; mu}; par = (lambda, mu)."""
    lam, mu = par
    out = K_M(f, u, mu)
    if lam:
        out = add(out, mul(smul(lam, D_M(f)), u))
    return out


# =================================================================== gradings

def degL(a, S):
    """deg t = 2, deg w_i = 1."""
    return 2 * int(a) + len(S)


def degM(a, S):
    """deg t = 2, deg zeta = 0, deg theta = deg xi = deg eta = 1."""
    return 2 * int(a) + len([i for i in S if i != MZETA])


# =============================================================== verification

def keys(arange, subsets=SUBSETS):
    return [(Q(a), S) for a in arange for S in subsets]


def jacobi_bad(bracket, ks):
    bad = 0
    for kf, kg, kh in product(ks, repeat=3):
        f, g, h = mono(*kf), mono(*kg), mono(*kh)
        lhs = bracket(f, bracket(g, h))
        rhs = add(bracket(bracket(f, g), h),
                  smul((-1) ** (parity(f) * parity(g)),
                       bracket(g, bracket(f, h))))
        if add(lhs, smul(-1, rhs)):
            bad += 1
    return bad


def module_bad(act, bracket, par, ks, us):
    """[K_f, K_g].u = K_{f,g}.u -- the module axiom."""
    bad = 0
    for kf, kg in product(ks, repeat=2):
        f, g = mono(*kf), mono(*kg)
        br = bracket(f, g)
        for ku in us:
            u = {ku: Q(1)}
            lhs = add(act(par, f, act(par, g, u)),
                      smul(-((-1) ** (parity(f) * parity(g))),
                           act(par, g, act(par, f, u))))
            rhs = act(par, br, u) if br else {}
            if add(lhs, smul(-1, rhs)):
                bad += 1
    return bad


# ===================================================== invariant bilinear forms

def diag_weights(act, par, bas, gen):
    """The eigenvalue of the (verified diagonal) even generator gen."""
    g = mono(*gen)
    out = {}
    for ku in bas:
        img = act(par, g, {ku: Q(1)})
        if set(img) - {ku}:
            raise ValueError("%s is not diagonal on %s: %s" % (gen, ku, img))
        out[ku] = img.get(ku, Q(0))
    return out


def _gens_default(amin=-2, amax=2):
    return keys(range(amin, amax + 1))


def _weight_shift(img, ws, inwin):
    """The weight vector by which K_f shifts, read off its action; None if
    K_f kills the whole window.  Verified to be the same for every monomial."""
    out = None
    for ku, im in img.items():
        for kp in im:
            if kp not in inwin:
                continue
            v = tuple(w[kp] - w[ku] for w in ws)
            if out is None:
                out = v
            elif out != v:
                raise ValueError("K_f is not a weight vector: %s vs %s"
                                 % (out, v))
    return out


def invariant_forms(act, par1, par2, N=4, gens=None, torus=TORUS_L,
                    verify_gens=None):
    """Basis of the space of invariant bilinear forms V1 x V2 -> C.

    The unknowns are cut down first by the DIAGONAL generators -- K_t and the
    maximal torus -- whose invariance equations are diagonal, hence exact and
    free; the remaining generators then give the linear system.  An equation is
    used only if every monomial it mentions lies inside the window, so no
    boundary term is silently dropped.
    """
    bas = keys(range(-N, N + 1))
    diag = [(Q(1), ())] + list(torus)
    w1 = [diag_weights(act, par1, bas, g) for g in diag]
    w2 = [diag_weights(act, par2, bas, g) for g in diag]

    # index the admissible pairs, bucketed by the weight vector of the 2nd slot
    idx, by_key = {}, {}
    for kv in bas:
        key = tuple(w[kv] for w in w2)
        by_key.setdefault(key, []).append(kv)
    for ku in bas:
        want = tuple(-w[ku] for w in w1)
        for kv in by_key.get(want, ()):
            idx[(ku, kv)] = len(idx)
    if not idx:
        return [], idx

    if gens is None:
        gens = _gens_default()
    inwin = set(bas)
    rows = []
    for kf in gens:
        f = mono(*kf)
        pf = parity(f)
        img1 = {ku: act(par1, f, {ku: Q(1)}) for ku in bas}
        wf = _weight_shift(img1, w1, inwin)
        if wf is None:
            continue                        # K_f acts by zero on the window
        img2 = {kv: act(par2, f, {kv: Q(1)}) for kv in bas}
        for ku in bas:
            want = tuple(-a - b for a, b in zip(wf, [w[ku] for w in w1]))
            for kv in by_key.get(want, ()):
                row = {}
                ok = True
                for kp, co in img1[ku].items():
                    if kp not in inwin:
                        ok = False
                        break
                    j = idx.get((kp, kv))
                    if j is not None:
                        row[j] = row.get(j, Q(0)) + co
                if not ok:
                    continue
                sgn = (-1) ** (pf * (len(ku[1]) % 2))
                for kp, co in img2[kv].items():
                    if kp not in inwin:
                        ok = False
                        break
                    j = idx.get((ku, kp))
                    if j is not None:
                        row[j] = row.get(j, Q(0)) + sgn * co
                if not ok:
                    continue
                row = {j: c for j, c in row.items() if c}
                if row:
                    rows.append(row)
    if not rows:
        dense = [[Q(0)] * len(idx)]
    else:
        seen, dense = set(), []
        for row in rows:
            sig = tuple(sorted(row.items()))
            if sig in seen:
                continue
            seen.add(sig)
            d = [Q(0)] * len(idx)
            for j, c in row.items():
                d[j] = c
            dense.append(d)
    ns = nullspace(dense)
    inv = {v: k for k, v in idx.items()}
    out = []
    for vec in ns:
        d = {}
        for j, c in enumerate(vec):
            if c:
                d[inv[j]] = Q(c)
        if d:
            out.append(d)
    if verify_gens:
        for form in out:
            bad = check_invariance(act, par1, par2, form, verify_gens, N)
            if bad:
                raise ValueError("a 'solution' is not invariant: %d defects"
                                 % bad)
    return out, idx


def check_invariance(act, par1, par2, form, gens, N):
    """B(f.u, v) + (-1)^{p(f)p(u)} B(u, f.v) = 0 on every pair in the window."""
    bas = keys(range(-N + 2, N - 1))
    bad = 0
    for kf in gens:
        f = mono(*kf)
        pf = parity(f)
        for ku in bas:
            iu = act(par1, f, {ku: Q(1)})
            for kv in bas:
                iv = act(par2, f, {kv: Q(1)})
                tot = Q(0)
                for kp, co in iu.items():
                    tot += co * form.get((kp, kv), Q(0))
                sgn = (-1) ** (pf * (len(ku[1]) % 2))
                for kp, co in iv.items():
                    tot += sgn * co * form.get((ku, kp), Q(0))
                if tot:
                    bad += 1
    return bad


def form_report(form, N):
    """(parity, the set of symmetry ratios, non-degeneracy, the dead modes)."""
    pars = {(len(ku[1]) + len(kv[1])) % 2 for (ku, kv) in form}
    par = pars.pop() if len(pars) == 1 else None
    eps = set()
    for (ku, kv), c in form.items():
        c2 = form.get((kv, ku), Q(0))
        sg = (-1) ** ((len(ku[1]) % 2) * (len(kv[1]) % 2))
        if c2 == 0:
            eps.add("degenerate-pair")
        else:
            eps.add(Q(c) / (sg * c2))
    inner = keys(range(-N + 2, N - 1))
    full = keys(range(-N, N + 1))
    dead = [ku for ku in inner
            if not any(form.get((ku, kv), Q(0)) for kv in full)]
    dead += [kv for kv in inner
             if not any(form.get((ku, kv), Q(0)) for ku in full)]
    return par, eps, (not dead), dead


def split_symmetry(forms):
    """Split a space of even forms into its supersymmetric and its
    anti-supersymmetric parts (Spin-, resp. Osc-admissible)."""
    sym, asym = [], []
    for form in forms:
        tr = {}
        for (ku, kv), c in form.items():
            sg = (-1) ** ((len(ku[1]) % 2) * (len(kv[1]) % 2))
            tr[(kv, ku)] = tr.get((kv, ku), Q(0)) + sg * c
        s = {k: (form.get(k, Q(0)) + tr.get(k, Q(0))) / 2
             for k in set(form) | set(tr)}
        a = {k: (form.get(k, Q(0)) - tr.get(k, Q(0))) / 2
             for k in set(form) | set(tr)}
        s, a = clean(s), clean(a)
        if s:
            sym.append(s)
        if a:
            asym.append(a)
    return _indep(sym), _indep(asym)


def _indep(fs):
    """Keep a maximal independent subset."""
    out, keysall = [], []
    for f in fs:
        cand = out + [f]
        cols = sorted({k for g in cand for k in g})
        M = [[g.get(k, Q(0)) for k in cols] for g in cand]
        if exrank(M) == len(cand):
            out.append(f)
    return out


# ================================================== the Fock space and the lift

class Fock:
    """Spin (shift = 1) or Osc (shift = 0) built on a space with a pairing.

    modes    -- an ordered list of mode labels;
    par      -- label -> parity of the corresponding vector of V;
    Bfun     -- (m, n) -> B(m, n);
    actfun   -- (f, m) -> {label: coefficient}, the g-action;
    order    -- label -> a sortable key; the smaller half of each sigma-orbit
                is declared creation.
    """

    def __init__(self, modes, par, Bfun, actfun, order, shift):
        self.allmodes = list(modes)
        self.par_ = par
        self.B = Bfun
        self.act = actfun
        self.order = order
        self.shift = shift
        self._sig = {}
        for m in self.allmodes:
            hits = [n for n in self.allmodes if self.B(m, n)]
            if len(hits) != 1:
                raise ValueError("sigma is not an involution at %s: %d partners"
                                 % (m, len(hits)))
            self._sig[m] = hits[0]
        for m in self.allmodes:
            if self._sig[m] == m:
                raise ValueError("self-dual mode %s: the 1/sqrt2 is needed" % m)

    def sigma(self, m):
        return self._sig[m]

    def beta(self, m):
        return self.B(self.sigma(m), m)

    def tp(self, m):
        return (self.par_(m) + self.shift) % 2

    def is_creation(self, m):
        return self.order(m) < self.order(self._sig[m])

    def modes(self):
        return self.allmodes


VAC = {(): Q(1)}


def sadd(a, b, s=Q(1)):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Q(0)) + s * v
        if out[k] == 0:
            del out[k]
    return out


def psi(W, m, st):
    if W.is_creation(m):
        return _psi_create(W, m, st)
    return _psi_annih(W, m, st)


def _psi_create(W, m, st):
    out = {}
    tm = W.tp(m)
    for word, co in st.items():
        pos, sgn = 0, 1
        while pos < len(word) and W.order(word[pos]) < W.order(m):
            if tm and W.tp(word[pos]):
                sgn = -sgn
            pos += 1
        if tm and pos < len(word) and word[pos] == m:
            continue                        # fermionic square
        new = word[:pos] + (m,) + word[pos:]
        out[new] = out.get(new, Q(0)) + sgn * co
    return {k: v for k, v in out.items() if v != 0}


def _psi_annih(W, m, st):
    out = {}
    tm = W.tp(m)
    for word, co in st.items():
        sgn = 1
        for k in range(len(word)):
            nk = word[k]
            b = W.B(m, nk)
            if b:
                rest = word[:k] + word[k + 1:]
                out[rest] = out.get(rest, Q(0)) + sgn * b * co
            if tm and W.tp(nk):
                sgn = -sgn
    return {k: v for k, v in out.items() if v != 0}


def hatX(W, f, st):
    """the quadratic lift of eq. (150), 1/2 sum_alpha psi_{X e_alpha} e^alpha."""
    tot = {}
    for m in W.allmodes:
        b = W.beta(m)
        if b == 0:
            continue
        img = W.act(f, m)
        if not img:
            continue
        part = psi(W, W.sigma(m), st)
        if not part:
            continue
        for p_, co in img.items():
            if p_ not in W._sig:
                continue                      # outside the window: dropped
            q = psi(W, p_, part)
            if q:
                tot = sadd(tot, q, Q(1, 2) * co / b)
    return tot


def test_hat(W, fs, ms):
    """[hat X, psi_v] = psi_{Xv} (super sense) -- the defining property."""
    bad = 0
    for f in fs:
        pf = parity(f)
        for m in ms:
            sgn = -((-1) ** (pf * W.tp(m)))
            lhs = sadd(hatX(W, f, psi(W, m, VAC)),
                       psi(W, m, hatX(W, f, VAC)), Q(sgn))
            rhs = {}
            for p_, co in W.act(f, m).items():
                if p_ in W._sig:
                    rhs = sadd(rhs, psi(W, p_, VAC), co)
            if sadd(lhs, rhs, Q(-1)):
                bad += 1
    return bad


def cocycle(W, bracket, f, g):
    """the scalar [hat f, hat g] - hat{f, g} on the vacuum (f, g even)."""
    lhs = sadd(hatX(W, f, hatX(W, g, VAC)), hatX(W, g, hatX(W, f, VAC)), Q(-1))
    br = bracket(f, g)
    if br:
        lhs = sadd(lhs, hatX(W, br, VAC), Q(-1))
    return lhs.get((), Q(0)), {k: v for k, v in lhs.items() if k != ()}


# =================================== the divergence, and the module Vol
#
# Div(sum f_i d_i) = sum (-1)^{p(f_i) p(d_i)} d_i(f_i), with p(d_t) = 0 and
# p(d_w) = 1.  The module Vol is the rank-1 module whose multiplier is
# Div K_f, so comparing Div with lambda (d_t or cD)(f) + mu t^{-1} nabla(f)
# -- the multipliers of alpha_1^{lambda/2}, resp. tilde alpha^{lambda/2}, and
# of the twist t^mu -- identifies Vol inside the family cF_{lambda; mu}.

def comps_L(f):
    p = parity(f)
    out = {"t": nabla(f)}
    for (x, y) in LPAIRS:
        out[x] = smul(-((-1) ** p),
                      add(d_odd(f, y), smul(-1, mul(mono(0, (x,)), d_t(f)))))
        out[y] = smul(-((-1) ** p),
                      add(d_odd(f, x), smul(-1, mul(mono(0, (y,)), d_t(f)))))
    return out


def comps_M(f):
    p = parity(f)
    s = (-1) ** p
    out = {"t": nabla(f)}
    out[MZETA] = add(smul(Q(-1, 2), mul(nabla(f), mono(-1, (MZETA,)))),
                     smul(s, mul(mono(-1), d_odd(f, MZETA))),
                     mul(D_M(f), mono(0, (MZETA,))))
    out[MTHETA] = add(smul(s, d_odd(f, MTHETA)),
                      mul(D_M(f), mono(0, (MTHETA,))))
    for (x, y) in MPAIRS:
        out[x] = add(smul(s, d_odd(f, y)), mul(D_M(f), mono(0, (x,))))
        out[y] = add(smul(s, d_odd(f, x)), mul(D_M(f), mono(0, (y,))))
    return out


def divergence(comps):
    out = d_t(comps["t"])
    for i in range(1, NODD + 1):
        c = comps.get(i, {})
        if c:
            out = add(out, smul((-1) ** parity(c), d_odd(c, i)))
    return out


def vol_params(compfun, dfun, ks):
    """(lambda, mu) with Div K_f = lambda dfun(f) + mu t^{-1} nabla(f)."""
    rows, rhs = [], []
    for k in ks:
        f = mono(*k)
        D = divergence(compfun(f))
        A = dfun(f)
        B = mul(mono(-1), nabla(f))
        cols = sorted(set(D) | set(A) | set(B))
        for c in cols:
            rows.append([A.get(c, Q(0)), B.get(c, Q(0))])
            rhs.append(D.get(c, Q(0)))
    from exactlin import solve as _solve
    x = _solve(rows, rhs)
    if x is None:
        return None
    lam, mu = Q(x[0]), Q(x[1])
    for k in ks:
        f = mono(*k)
        want = add(smul(lam, dfun(f)), smul(mu, mul(mono(-1), nabla(f))))
        if add(divergence(compfun(f)), smul(-1, want)):
            return None
    return lam, mu
