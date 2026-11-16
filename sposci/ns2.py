#!/usr/bin/env python3
r"""
ns2.py -- TWO odd indeterminates: the analogs of Proposition 7.3 and of
Spin(V (+) V^#) for

        k^L(1|2)   (Neveu-Schwarz, untwisted)   with  V = cF_{a,b;mu},
        k^M(1|2)   (Ramond, twisted)            with  V = cF_{a;mu}.

This answers the question:

  "In a similar way, please consider Spin(V (+) V^*), where V = cF_{a,b;mu}
   for k^L(1|2) and V = cF_{a;mu} for k^M(1|2)."

It is the sequel of ns1.py, which did the same for one odd
indeterminate.  Everything is exact rational arithmetic; nothing is assumed:
the bracket of each algebra, the module axiom, the multiplier of [d xi], the
invariance of every form found and the Clifford-Weyl relations are all verified
before use.

--------------------------------------------------------------------- set-up

Functions.  R(2) = C[t,t^{-1}] (x) Lambda(w_1, w_2); a monomial is a key
(a, S) with a in Q and S a sorted subtuple of (1, 2), standing for t^a w^S.

  * k^L(1|2): the xi-eta realization, w_1 = xi, w_2 = eta, so that
    alpha_1 = dt + xi d eta + eta d xi, see eq. (40)/(41) of the text.
    This is the realization in which the SECOND parameter of the weighted
    densities lives, [d xi] being the class of d xi mod cF alpha_1.

  * k^M(1|2): n = 2 = 2k+2 with k = 0, so the odd indeterminates are the
    untwisted theta = w_1 and the twisted zeta = w_2, see eq. equation 53.

k^L(1|2), eqs. (3a) and (4a) of the text:
    K_f = nabla(f) d_t - (-1)^{p(f)} [ (d_eta f - xi d_t f) d_xi
                                     + (d_xi  f - eta d_t f) d_eta ],
    {f,g} = nabla(f) d_t g - d_t f nabla(g)
            - (-1)^{p(f)} ( d_xi f d_eta g + d_eta f d_xi g ),
with nabla = 2 - cE, cE = xi d_xi + eta d_eta.

k^M(1|2), eqs. (K^M), (K^Ma), (10a), (12a):
    cD = d_t - (zeta/2t) d_zeta,   nabla = 2 - cE,
    K^M_f = nabla(f) cD + H^M_f + cD(f) cE,
    H^M_f = (-1)^{p(f)} ( cz (1/t) d_zeta f d_zeta + ct d_theta f d_theta ),
    {f,g}_R = nabla(f) cD(g) - cD(f) nabla(g)
              + (-1)^{p(f)} ( cz (1/t) d_zeta f d_zeta g
                            + ct d_theta f d_theta g ),
the two constants (cz, ct) being fixed below by the Jacobi identity and by
[K_f, K_g] = K_{{f,g}} rather than read off the printed formulas.

Modules.
  * Over k^L(1|2):  cF_{lambda, nu; mu} := cF_{lambda; mu} [d xi]^nu, see
    eq. equation 97.  The multiplier of alpha_1 is 2 d_t(f), hence that of
    alpha_1^{lambda/2} is lambda d_t(f); the multiplier of [d xi] is computed
    from scratch in dxi_multiplier() below and comes out
            m(f) = d_t f - d_xi d_eta f = (d_t + Xi)(f),   Xi := -d_xi d_eta.
    So  K_f . u = K_f(u) + ( (lambda + nu) d_t f + nu Xi(f) ) u.
    In particular K_{xi eta} acts on the generator by nu -- which is the
    label H of the tables of Subsection (7.4.7).

  * Over k^M(1|2):  cF^M_{lambda; mu} := cF^M_{0;mu} (tilde alpha)^{lambda/2},
    the multiplier of tilde alpha being 2 cD(f), so
            K^M_f . u = K^M_f(u) + lambda cD(f) u.
    There is no second parameter here; the reason is Proposition prTwoZero
    of the text: k^M(1|2)_0 = <K_t, K_{t zeta}> has sdim (1|1) and
    {t zeta, t zeta}_{R.b.} = -t, so it is the odd Heisenberg superalgebra,
    whose irreducible modules form ONE one-parameter family, whereas
    k^L(1|2)_0 = <K_t, K_{xi eta}> is commutative and purely even, whence two.
"""

from fractions import Fraction as Q
from itertools import product
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))
from exactlin import nullspace                            # noqa: E402

XI, ETA = 1, 2                      # k^L(1|2): w_1 = xi, w_2 = eta
THETA, ZETA = 1, 2                  # k^M(1|2): w_1 = theta, w_2 = zeta
SUBSETS = ((), (1,), (2,), (1, 2))


# ============================================================ the algebra R(2)

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


def _sortsign(S):
    """Sort a tuple of odd indices; return (sorted, sign), or (None, 0) if
    an index repeats (the monomial is then 0)."""
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


# =============================================================== k^L(1|2)

def br_L(f, g):
    res = add(mul(nabla(f), d_t(g)), smul(-1, mul(d_t(f), nabla(g))))
    odd = add(mul(d_odd(f, XI), d_odd(g, ETA)), mul(d_odd(f, ETA), d_odd(g, XI)))
    return add(res, smul(-((-1) ** parity(f)), odd))


def K_L(f, u, mu=Q(0)):
    """the contact vector field K_f applied to a function u in t^mu R(2)."""
    p = parity(f)
    out = mul(nabla(f), d_t(u, mu))
    cxi = add(d_odd(f, ETA), smul(-1, mul(mono(0, (XI,)), d_t(f))))
    ceta = add(d_odd(f, XI), smul(-1, mul(mono(0, (ETA,)), d_t(f))))
    out = add(out, smul(-((-1) ** p),
                        add(mul(cxi, d_odd(u, XI)), mul(ceta, d_odd(u, ETA)))))
    return out


def act_L(par, f, u):
    """K_f . u  on cF_{lambda, nu; mu};  par = (lambda, nu, mu)."""
    lam, nu, mu = par
    out = K_L(f, u, mu)
    mult = d_t(f) if lam else {}
    mult = add(smul(lam, mult), smul(nu, dxi_multiplier(f)) if nu else {})
    return add(out, mul(mult, u)) if mult else out


# =============================================================== k^M(1|2)

def D_M(f, mu=Q(0)):
    """cD = d_t - (zeta/2t) d_zeta."""
    return add(d_t(f, mu),
               smul(Q(-1, 2), mul(mono(-1, (ZETA,)), d_odd(f, ZETA))))


def br_M(f, g, cz=Q(1), ct=Q(1)):
    res = add(mul(nabla(f), D_M(g)), smul(-1, mul(D_M(f), nabla(g))))
    mp = add(smul(cz, mul(mono(-1), mul(d_odd(f, ZETA), d_odd(g, ZETA)))),
             smul(ct, mul(d_odd(f, THETA), d_odd(g, THETA))))
    return add(res, smul((-1) ** parity(f), mp))


def K_M(f, u, mu=Q(0), cz=Q(1), ct=Q(1)):
    p = parity(f)
    out = mul(nabla(f), D_M(u, mu))
    mp = add(smul(cz, mul(mono(-1), mul(d_odd(f, ZETA), d_odd(u, ZETA)))),
             smul(ct, mul(d_odd(f, THETA), d_odd(u, THETA))))
    out = add(out, smul((-1) ** p, mp))
    return add(out, mul(D_M(f), cE(u)))


def act_M(par, f, u, cz=Q(1), ct=Q(1)):
    """K^M_f . u  on cF^M_{lambda; mu};  par = (lambda, mu)."""
    lam, mu = par
    out = K_M(f, u, mu, cz, ct)
    return add(out, mul(smul(lam, D_M(f)), u))


# ============================================ the multiplier of [d xi], checked
#
# 1-forms are written in the basis dt, d xi, d eta with function coefficients;
# L_{K_f} d w = d (K_f(w)).  Modulo cF alpha_1, dt = -xi d eta - eta d xi.

def d_odd_R(f, i):
    """right derivative d/dw_i."""
    out = {}
    for (a, S), v in f.items():
        if i in S:
            k = S.index(i)
            T = S[:k] + S[k + 1:]
            out[(a, T)] = out.get((a, T), Q(0)) + ((-1) ** (len(S) - 1 - k)) * v
    return clean(out)


# ------------------------------------------------ 1-forms, and the class [d xi]
#
# A 1-form is a dict {var: coefficient}, var in ("t", XI, ETA), the coefficient
# standing on the LEFT: omega = omega_t dt + omega_xi d xi + omega_eta d eta,
# with p(dx) = p(x), so that d is even.  Then d g = sum_j (d^R_j g) dx^j with
# RIGHT derivatives, and, K_f being the field with components comps(f),
#     L_{K_f}(omega) = sum_i [ K_f(omega_i) dx^i
#                            + (-1)^{p(f) p(omega_i)} omega_i d(g_i) ].
# Two identities fix this convention, and both are verified in runns2.py:
#   (a) L_{K_f}(alpha) = 2 d_t(f) alpha  for  alpha = dt - xi d eta - eta d xi,
#       which is the text's alpha_1 of eq. (40) written with LEFT coefficients
#       (d eta . xi = - xi d eta, both factors being odd);
#   (b) L_{[K_f, K_g]} = [L_{K_f}, L_{K_g}] on 1-forms.

FVARS = ("t", XI, ETA)
ALPHA1 = {"t": mono(0), XI: mono(0, (ETA,), -1), ETA: mono(0, (XI,), -1)}


def _dR(f, i):
    return d_t(f) if i == "t" else d_odd_R(f, i)


def comps(f):
    """the components of K_f along d/dt, d/d xi, d/d eta."""
    p = parity(f)
    return {"t": nabla(f),
            XI: smul(-((-1) ** p),
                     add(d_odd(f, ETA), smul(-1, mul(mono(0, (XI,)), d_t(f))))),
            ETA: smul(-((-1) ** p),
                      add(d_odd(f, XI), smul(-1, mul(mono(0, (ETA,)), d_t(f)))))}


def Lie1(f, om):
    """L_{K_f} on a 1-form."""
    p = parity(f)
    g = comps(f)
    out = {v: {} for v in FVARS}
    for i in FVARS:
        for km, cm in om.get(i, {}).items():
            wi = {km: cm}
            s = (-1) ** (p * (len(km[1]) % 2))
            out[i] = add(out[i], K_L(f, wi))
            for j in FVARS:
                out[j] = add(out[j], smul(s, mul(wi, _dR(g[i], j))))
    return {k: clean(v) for k, v in out.items()}


def reduce_mod_alpha(om):
    """the class of a 1-form in Omega^1 / cF alpha_1: (c_[d xi], c_[d eta])."""
    wt = om.get("t", {})
    return (add(om.get(XI, {}), mul(wt, mono(0, (ETA,)))),
            add(om.get(ETA, {}), mul(wt, mono(0, (XI,)))))


D_XI = {"t": {}, XI: mono(0), ETA: {}}


def dxi_multiplier(f):
    """the multiplier m(f) of the rank-1 module cF[d xi]:
    L_{K_f}[d xi] = m(f) [d xi], the [d eta] component being 0."""
    cx, cy = reduce_mod_alpha(Lie1(f, D_XI))
    if cy:
        raise ValueError("cF[d xi] is not a submodule at f = %s: %s" % (f, cy))
    return cx


# =================================== the rank-1 module structures, from scratch
#
# A rank-1 module on t^mu R(2) is  f . u = K_f(u) + c(f) u  with c a 1-cocycle:
#       c({f, g}) = K_f(c(g)) - (-1)^{p(f) p(g)} K_g(c(f)),
# and two cocycles differing by a coboundary c(f) = K_f(u), u in the degree-0
# part of R(2), give isomorphic modules.  Both spaces are computed below; the
# quotient is the number of parameters of the family.

def degL(a, S):
    """the k^L(1|2) grading: deg t = 2, deg xi = deg eta = 1."""
    return 2 * int(a) + len(S)


def degM(a, S):
    """the k^M(1|2) grading: deg t = 2, deg theta = 1, deg zeta = 0."""
    return 2 * int(a) + (1 if THETA in S else 0)


def _fbasis(delta, N, deg=degL):
    """monomials t^a w^S of the given grading, |a| <= N."""
    return [(Q(a), S) for a in range(-N, N + 1) for S in SUBSETS
            if deg(a, S) == delta]


def multiplier_space(Kfun, bracket, N=4, D=3, deg=degL):
    """(dim cocycles, dim coboundaries, basis of cocycles) in grading 0.

    Kfun(f, u) is the vector field K_f applied to the function u.
    Unknowns: c(f) in R(2)_{deg f - 2} for every monomial f of grading
    |deg| <= D + 2; equations: the cocycle identity for every pair (f, g)
    of monomials whose bracket stays inside the window.
    """
    fb = {}                                   # grading -> basis of R(2)_grading
    for d in range(-2 * N, 2 * N + 1):
        fb[d] = _fbasis(d, N, deg)
    gens = [k for d in range(-D, D + 3) for k in _fbasis(d, N, deg)]
    idx, spots = {}, {}
    for kf in gens:
        d = deg(*kf) - 2
        spots[kf] = []
        for ku in fb.get(d, []):
            idx[(kf, ku)] = len(idx)
            spots[kf].append(ku)

    def crow(kf, coef, row, sign=1):
        """add sign*coef*c(kf) into row; returns False if kf is off-window."""
        if kf not in spots:
            return False
        for ku in spots[kf]:
            row[idx[(kf, ku)]] += 0          # touch
        return True

    rows = []
    for kf in gens:
        for kg in gens:
            f, g = mono(*kf), mono(*kg)
            br = bracket(f, g)
            if any(k not in spots for k in br):
                continue
            pf, pg = parity(f), parity(g)
            # one equation per monomial of the value space
            dval = deg(*kf) + deg(*kg) - 4
            targets = fb.get(dval, [])
            if not targets:
                continue
            tpos = {k: i for i, k in enumerate(targets)}
            eqs = [[Q(0)] * len(idx) for _ in targets]
            ok = True
            # - c({f,g})
            for k, co in br.items():
                if k not in spots:
                    ok = False
                    break
                for ku in spots[k]:
                    if ku in tpos:
                        eqs[tpos[ku]][idx[(k, ku)]] -= co
                    else:
                        ok = False
            if not ok:
                continue
            # + K_f(c(g))
            for ku in spots.get(kg, []):
                img = Kfun(f, {ku: Q(1)})
                for k2, co in img.items():
                    if k2 not in tpos:
                        ok = False
                        break
                    eqs[tpos[k2]][idx[(kg, ku)]] += co
            # - (-1)^{pf pg} K_g(c(f))
            sg = (-1) ** (pf * pg)
            for ku in spots.get(kf, []):
                img = Kfun(g, {ku: Q(1)})
                for k2, co in img.items():
                    if k2 not in tpos:
                        ok = False
                        break
                    eqs[tpos[k2]][idx[(kf, ku)]] -= sg * co
            if not ok:
                continue
            for e in eqs:
                if any(e):
                    rows.append(e)
    if not rows:
        rows = [[Q(0)] * len(idx)]
    coc = nullspace(rows)
    # coboundaries: c(f) = K_f(u) for u in R(2)_0
    cob = []
    for ku in fb.get(0, []):
        vec = [Q(0)] * len(idx)
        ok = True
        for kf in gens:
            img = Kfun(mono(*kf), {ku: Q(1)})
            for k2, co in img.items():
                if (kf, k2) not in idx:
                    ok = False
                    break
                vec[idx[(kf, k2)]] += co
        if ok and any(vec):
            cob.append(vec)
    from exactlin import rank as _rank
    dc = _rank(coc) if coc else 0
    dcob = _rank(cob) if cob else 0
    return dc, dcob, coc, idx


# =============================================================== verification

def keys(arange, subsets=SUBSETS):
    return [(Q(a), S) for a in arange for S in subsets]


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


def module_bad(act, bracket, par, ks, us):
    """[K_f, K_g].u = K_{f,g}.u -- the module axiom."""
    bad = 0
    for kf, kg in product(ks, repeat=2):
        f, g = mono(*kf), mono(*kg)
        for ku in us:
            u = {ku: Q(1)}
            lhs = add(act(par, f, act(par, g, u)),
                      smul(-((-1) ** (parity(f) * parity(g))),
                           act(par, g, act(par, f, u))))
            rhs = act(par, bracket(f, g), u)
            if add(lhs, smul(-1, rhs)):
                bad += 1
    return bad


# ===================================================== invariant bilinear forms

def weights(act, par, bas):
    """The eigenvalue of K_t on each basis element (verified diagonal)."""
    t = mono(1)
    out = {}
    for ku in bas:
        img = act(par, t, {ku: Q(1)})
        if set(img) - {ku}:
            raise ValueError("K_t is not diagonal on %s: %s" % (ku, img))
        out[ku] = img.get(ku, Q(0))
    return out


def invariant_forms(act, par1, par2, N=5, gens=None):
    """Basis of the space of invariant bilinear forms V1 x V2 -> C.

    Only the pairs whose two K_t-weights add up to 0 are carried as unknowns;
    for a generator K_f of weight w, only the pairs of total weight -w give
    a non-trivial equation, and an equation is used only if every monomial it
    mentions lies inside the window, so no boundary term is silently dropped.
    """
    bas = keys(range(-N, N + 1))
    w1 = weights(act, par1, bas)
    w2 = weights(act, par2, bas)
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
    by_w2 = {}
    for kv in bas:
        by_w2.setdefault(w2[kv], []).append(kv)
    rows = []
    for kf in gens:
        f = mono(*kf)
        pf = parity(f)
        wf = None
        img1 = {}
        for ku in bas:
            im = act(par1, f, {ku: Q(1)})
            img1[ku] = im
            for kp in im:
                if kp in inwin:
                    wf = w1[kp] - w1[ku]
        if wf is None:
            continue
        img2 = {kv: act(par2, f, {kv: Q(1)}) for kv in bas}
        for ku in bas:
            for kv in by_w2.get(-wf - w1[ku], ()):
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
                if not ok or not any(row.values()):
                    continue
                dense = [Q(0)] * len(idx)
                for j, c in row.items():
                    dense[j] = c
                rows.append(dense)
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
    """(parity, symmetry, non-degeneracy on the inner window)."""
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
    inner = keys(range(-N + 3, N - 2))
    full = keys(range(-N, N + 1))
    dead = [ku for ku in inner
            if not any(form.get((ku, kv), Q(0)) for kv in full)]
    dead += [kv for kv in inner
             if not any(form.get((ku, kv), Q(0)) for ku in full)]
    return par, eps, (not dead), dead


# ======================================================= the doubled module W
#
# W = V (+) V^#; modes (s, a, S), s = 0 for V, s = 1 for V^#.

class Wspace:
    def __init__(self, actfun, par, par_d, pair, par_shift=0):
        self.act_ = actfun
        self.par, self.par_d = par, par_d
        self.pair = pair
        self.par_shift = par_shift
        self._sig = {}
        for (ku, kv), c in pair.items():
            if c:
                self._sig[(0,) + ku] = (1,) + kv
                self._sig[(1,) + kv] = (0,) + ku
        # the polarization constant: a + a' is the same for every paired couple
        sums = {ku[0] + kv[0] for (ku, kv), c in pair.items() if c}
        if len(sums) != 1:
            raise ValueError("the pairing does not have a constant t-degree: %s"
                             % sorted(map(str, sums)))
        self.Ksum = Q(sums.pop())
        if self.Ksum.denominator == 1 and self.Ksum % 2 == 0:
            raise ValueError("self-paired t-degree %s: polarization ambiguous"
                             % self.Ksum)

    def par_(self, m):
        return (len(m[2]) + (self.par_shift if m[0] == 1 else 0)) % 2

    def sigma(self, m):
        return self._sig[m]

    def B(self, m1, m2):
        (s1, a1, S1), (s2, a2, S2) = m1, m2
        if s1 == s2:
            return Q(0)
        if s1 == 0:
            return self.pair.get(((a1, S1), (a2, S2)), Q(0))
        return (-1) ** (self.par_(m1) * self.par_(m2)) * \
            self.pair.get(((a2, S2), (a1, S1)), Q(0))

    def beta(self, m):
        return self.B(self.sigma(m), m)

    def act(self, f, m):
        s, a, S = m
        u = {(a, S): Q(1)}
        p = self.par if s == 0 else self.par_d
        r = self.act_(p, f, u)
        return {(s, k[0], k[1]): v for k, v in r.items()}

    def is_creation(self, m):
        return 2 * m[1] < self.Ksum

    def modes(self, window):
        return [(s, Q(a), S) for s in (0, 1)
                for a in range(-window, window + 1) for S in SUBSETS]


# ------------------------------------------------- the Clifford-Weyl algebra
# psi_m has the SHIFTED parity tp(m) = par(m) + 1, and
#      psi_m psi_n = (-1)^{tp(m) tp(n)} psi_n psi_m + B(m, n).

VAC = {(): Q(1)}


def tp(W, m):
    return (W.par_(m) + 1) % 2


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
    tm = tp(W, m)
    for word, co in st.items():
        pos, sgn = 0, 1
        while pos < len(word) and word[pos] < m:
            if tm and tp(W, word[pos]):
                sgn = -sgn
            pos += 1
        if tm and pos < len(word) and word[pos] == m:
            continue                        # fermionic square
        new = word[:pos] + (m,) + word[pos:]
        out[new] = out.get(new, Q(0)) + sgn * co
    return {k: v for k, v in out.items() if v != 0}


def _psi_annih(W, m, st):
    out = {}
    tm = tp(W, m)
    for word, co in st.items():
        sgn = 1
        for k in range(len(word)):
            nk = word[k]
            b = W.B(m, nk)
            if b:
                rest = word[:k] + word[k + 1:]
                out[rest] = out.get(rest, Q(0)) + sgn * b * co
            if tm and tp(W, nk):
                sgn = -sgn
    return {k: v for k, v in out.items() if v != 0}


def hatX(W, f, st, window):
    tot = {}
    for m in W.modes(window):
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
            q = psi(W, p_, part)
            if q:
                tot = sadd(tot, q, Q(1, 2) * co / b)
    return tot


def test_hat(W, window, fs):
    """[hat X, psi_v] = psi_{Xv} (super sense) -- the defining property."""
    bad = 0
    for f in fs:
        pf = parity(f)
        for m in W.modes(2):
            sgn = -((-1) ** (pf * tp(W, m)))
            lhs = sadd(hatX(W, f, psi(W, m, VAC), window),
                       psi(W, m, hatX(W, f, VAC, window)), Q(sgn))
            rhs = {}
            for p_, co in W.act(f, m).items():
                rhs = sadd(rhs, psi(W, p_, VAC), co)
            if sadd(lhs, rhs, Q(-1)):
                bad += 1
    return bad


def cocycle(W, bracket, f, g, window):
    """the scalar [hat K_f, hat K_g] - hat K_{f,g} on the vacuum (f, g even)."""
    lhs = sadd(hatX(W, f, hatX(W, g, VAC, window), window),
               hatX(W, g, hatX(W, f, VAC, window), window), Q(-1))
    br = bracket(f, g)
    if br:
        lhs = sadd(lhs, hatX(W, br, VAC, window), Q(-1))
    return lhs.get((), Q(0)), {k: v for k, v in lhs.items() if k != ()}
