#!/usr/bin/env python3
r"""
vect2.py -- the analogs of Spin(V (+) V^#) and Spin(V) for

        vect^L(1|2)              with  V = Vol^a_mu,
        svect^L_lambda(1|2)      with  V = cF_mu  (= Vol^a_mu, see below).

This answers the question:

  "Now, please 'do the same' = compute highest weight for Spin(W) and Spin(V)
   for \fvect^L(1|2) and next \fsvect_lambda^L(1|2)"

It is the sequel of ns1.py, ns2.py
ns4.py and hw4.py, which did
the same for the contact superalgebras k^L(1|n), k^M(1|n).  Everything is exact
rational arithmetic; nothing is assumed: the bracket, the divergence, the module
axiom, the invariance of every form found and the Clifford-Weyl relations are
all verified before use.

----------------------------------------------------------------- the algebras

Functions: R(2) = C[t, t^{-1}] (x) Lambda(theta_1, theta_2); a monomial is
a key (a, S) with a in Q and S a sorted subtuple of (1, 2), standing for
t^a theta^S.  The ring routines are imported from ns2.py.

A vector field is a dict {var: function}, var in ("t", 1, 2), the coefficients
standing on the LEFT:

        D = f_t d_t + f_1 d_{theta_1} + f_2 d_{theta_2},   p(D) = p(f_t),

the derivatives being the LEFT ones.  Then D(uv) = D(u) v + (-1)^{p(D)p(u)}uD(v)
and the bracket of two fields is again a field, whose components are read off
its values on the generators t, theta_1, theta_2:

        [D, E](x) = D(E(x)) - (-1)^{p(D)p(E)} E(D(x)).

The divergence in this convention is

        Div D = d_t f_t + sum_i (-1)^{p(f_i)} d_{theta_i} f_i,

which is the one pinned by eq. equation 50 of the text:

        D in svect^L_lambda(1|2)  <==>  lambda f_t = -t Div D.

Both are verified in runvect2.py (Div [D, E] = D(Div E) -+ E(Div D) and the
basis eq. equation 77 of the text).

The Witt subalgebra is the text's own, eq. equation 77:

        LL_m := t^m ( t d_t + (lambda + m + 1)/2 (theta_1 d_1 + theta_2 d_2) ),

which lies in svect^L_lambda(1|2) for the given lambda and, for lambda = 0, is
one half of K_{t^{m+1}}: the standard Witt subalgebra of vect^L(1|2), the
grading operator being ell_0 = LL_0.  With the text's convention eq. equation 148,

        [L_a, L_b] = (b - a) L_{a+b} + (a^3 - a)/12 delta_{a,-b} z,

we have [LL_m, LL_n] = (n - m) LL_{m+n}, verified.

------------------------------------------------------------------ the modules

Vol^a_mu := t^mu R(2) vvol^a, of rank 1 over the functions, with

        L_D (u vvol^a) = ( D(u) + a Div(D) u ) vvol^a,

and p(vvol) = 2 mod 2 = 0 (the second convention of Subsection ssRes of the
text), so the parity of t^{mu+k} theta^S vvol^a is |S| mod 2.

  * over vect^L(1|2) the pair (a, mu) is a genuine 2-parameter family;
  * over svect^L_lambda(1|2) one has Vol^a_mu = cF_{0; mu - a lambda}, since
    Div D = -lambda f_t / t there, so the family has ONE parameter -- which is
    the analog of Proposition 7.3.9 of the text (g_0 = C LL_0 (+) sl(2) has
    a 1-parameter family of characters, sl(2) having none).

The Witt subalgebra of vect^L(1|2) is divergence-free, so it does not feel a at
all: over C[LL_m] the module Vol^a_mu is Vol^0_mu whatever a is.  Hence c and h
below do not depend on a; the labels H_i do.

------------------------------------------------------------------- the pairing

The residue eq. (99) of the text pairs t^{mu+k} theta^S vvol^a with
t^{mu'+k'} theta^{S^c} vvol^{a'} when a + a' = 1 and mu + mu' + k + k' = -1;
it is even and supersymmetric.  Nothing of this is assumed: invariant_forms()
below solves for the invariant bilinear forms from scratch.

-------------------------------------------------------------- the polarization

The energy is the ell_0-weight, NOT the t-degree: at n = 2 the fiber
Lambda(theta_1, theta_2) contributes |S|/2 to it, so the two orderings differ.
A mode is a creation operator iff its ell_0-weight is negative, which is the
convention of ns2.py/ns4.py (there the two coincide, K_t = 2 d_t killing the
thetas).  Modes of weight 0 come in sigma-couples {m, sigma(m)}; one member of
each has to be declared a creation operator by hand -- see remFourPolar of the
text -- and both choices are run below.
"""

from fractions import Fraction as Q
from itertools import product
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))

from ns2 import (SUBSETS, clean, add, smul, mul, mono, d_t, d_odd,   # noqa: E402
                 parity, keys, VAC, sadd, psi, hatX, tp)
from exactlin import nullspace, rank as exrank                       # noqa: E402

VARS = ("t", 1, 2)


# ================================================================ vector fields

def VF(ft=None, f1=None, f2=None):
    return {"t": clean(ft or {}), 1: clean(f1 or {}), 2: clean(f2 or {})}


def vclean(D):
    return {v: clean(D.get(v, {})) for v in VARS}


def vadd(*Ds):
    out = {v: {} for v in VARS}
    for D in Ds:
        for v in VARS:
            out[v] = add(out[v], D.get(v, {}))
    return out


def vsmul(c, D):
    return {v: smul(c, D.get(v, {})) for v in VARS}


def vzero(D):
    return not any(D.get(v) for v in VARS)


def vparity(D):
    """p(D) = p(f_t) = p(f_i) + 1."""
    ps = set()
    if D.get("t"):
        ps.add(parity(D["t"]))
    for i in (1, 2):
        if D.get(i):
            ps.add((parity(D[i]) + 1) % 2)
    if len(ps) > 1:
        raise ValueError("inhomogeneous vector field: %s" % D)
    return ps.pop() if ps else 0


def apply_vf(D, u, mu=Q(0)):
    """D(u) for u in t^mu R(2)."""
    out = mul(D.get("t", {}), d_t(u, mu))
    for i in (1, 2):
        if D.get(i):
            out = add(out, mul(D[i], d_odd(u, i)))
    return out


def br_vf(D, E):
    """[D, E], read off its values on the generators t, theta_1, theta_2."""
    s = (-1) ** (vparity(D) * vparity(E))
    return {v: add(apply_vf(D, E.get(v, {})),
                   smul(-s, apply_vf(E, D.get(v, {})))) for v in VARS}


def div(D):
    """Div D = d_t f_t + sum_i (-1)^{p(f_i)} d_{theta_i} f_i."""
    out = d_t(D.get("t", {}))
    for i in (1, 2):
        f = D.get(i, {})
        if f:
            out = add(out, smul((-1) ** parity(f), d_odd(f, i)))
    return out


def vname(D):
    def fn(f):
        parts = []
        for (a, S), c in sorted(f.items()):
            m = ("t^%s" % a if a else "") + "".join("w%d" % i for i in S)
            parts.append(("%s%s" % (c if c != 1 else "", m)) or "1")
        return "+".join(parts) if parts else "0"
    bits = []
    for v in VARS:
        if D.get(v):
            bits.append("(%s)d_%s" % (fn(D[v]), v))
    return " + ".join(bits) if bits else "0"


# ------------------------------------------------------- monomial bases of vect

def vect_basis(N):
    """all t^a theta^S d_v with |a| <= N."""
    out = []
    for a in range(-N, N + 1):
        for S in SUBSETS:
            for v in VARS:
                out.append(VF(**{{"t": "ft", 1: "f1", 2: "f2"}[v]: mono(a, S)}))
    return out


def deg_vf(D):
    """the doubled grading: deg t = 2, deg theta_i = 1."""
    ds = set()
    for v in VARS:
        for (a, S) in D.get(v, {}):
            ds.add(2 * int(a) + len(S) - (2 if v == "t" else 1))
    if len(ds) > 1:
        raise ValueError("inhomogeneous degree: %s" % vname(D))
    return ds.pop() if ds else None


# ==================================================== the two algebras' bases

def ell(m, lam=Q(0)):
    """the text's LL_m of eq. equation 77; for lam = 0 it is (1/2) K_{t^{m+1}}."""
    half = Q(lam + m + 1, 1) / 2
    return VF(ft=mono(m + 1),
              f1=smul(half, mono(m, (1,))),
              f2=smul(half, mono(m, (2,))))


def Sfield(m, j, lam=Q(0)):
    """the text's S^j_m of eq. equation 77."""
    k = Q(lam + m + 1)
    other = 2 if j == 1 else 1
    D = VF(ft=mono(m + 1, (j,)))
    #  t^m theta_j (theta_1 d_1 + theta_2 d_2) = t^m theta_j theta_other d_other
    D[other] = smul(k, mul(mono(m, (j,)), mono(0, (other,))))
    return vclean(D)


def svect_basis(m, lam):
    """the text's basis eq. equation 77 in the doubled degrees 2m (even) and
    2m + 1 (odd), plus theta_1 theta_2 d_t when lambda = 0 and m = 0."""
    ev = [("LL_%d" % m, ell(m, lam)),
          ("t^%d w1 d_2" % m, VF(f2=mono(m, (1,)))),
          ("t^%d w2 d_1" % m, VF(f1=mono(m, (2,)))),
          ("t^%d (w1 d_1 - w2 d_2)" % m,
           VF(f1=mono(m, (1,)), f2=smul(-1, mono(m, (2,)))))]
    if lam == 0 and m == 0:
        ev.append(("w1 w2 d_t", VF(ft=mono(0, (1, 2)))))
    od = [("S^1_%d" % m, Sfield(m, 1, lam)),
          ("S^2_%d" % m, Sfield(m, 2, lam)),
          ("t^%d d_1" % (m + 1), VF(f1=mono(m + 1))),
          ("t^%d d_2" % (m + 1), VF(f2=mono(m + 1)))]
    return ev, od


def svect_kernel(m, lam, parity_wanted):
    """A basis of {D of doubled degree d | lambda f_t + t Div D = 0}, computed
    from scratch; d = 2m for parity 0 and 2m+1 for parity 1."""
    d = 2 * m + (1 if parity_wanted else 0)
    bas = []
    for a in range(m - 3, m + 4):
        for S in SUBSETS:
            for v in VARS:
                D = VF(**{{"t": "ft", 1: "f1", 2: "f2"}[v]: mono(a, S)})
                if deg_vf(D) == d and vparity(D) == parity_wanted:
                    bas.append(D)
    # unknown coefficients x_i; the condition is linear
    tgt = {}
    rows = []
    cols = []
    for D in bas:
        val = add(smul(lam, D.get("t", {})), mul(mono(1), div(D)))
        cols.append(val)
        for k in val:
            tgt.setdefault(k, len(tgt))
    M = [[Q(0)] * len(bas) for _ in tgt]
    for j, val in enumerate(cols):
        for k, c in val.items():
            M[tgt[k]][j] = c
    if not M:
        M = [[Q(0)] * len(bas)]
    ns = nullspace(M)
    out = []
    for vec in ns:
        D = {v: {} for v in VARS}
        for j, c in enumerate(vec):
            if c:
                D = vadd(D, vsmul(c, bas[j]))
        if not vzero(D):
            out.append(D)
    return out, bas


# ===================================================================== modules
#
#   par = (a, mu):  V = Vol^a_mu = t^mu R(2) vvol^a.

def act(par, D, u):
    a, mu = par
    out = apply_vf(D, u, mu)
    if a:
        dv = div(D)
        if dv:
            out = add(out, smul(a, mul(dv, u)))
    return out


def modpar(k):
    """the parity of the basis monomial t^{mu+k} theta^S vvol^a."""
    return len(k[1]) % 2


def module_bad(par, fields, us):
    """[L_D, L_E] = L_{[D,E]} -- the module axiom."""
    bad = 0
    for D, E in product(fields, repeat=2):
        s = (-1) ** (vparity(D) * vparity(E))
        for ku in us:
            u = {ku: Q(1)}
            lhs = add(act(par, D, act(par, E, u)),
                      smul(-s, act(par, E, act(par, D, u))))
            rhs = act(par, br_vf(D, E), u)
            if add(lhs, smul(-1, rhs)):
                bad += 1
    return bad


def jacobi_bad(fields):
    bad = 0
    for D, E, F in product(fields, repeat=3):
        s = (-1) ** (vparity(D) * vparity(E))
        lhs = br_vf(D, br_vf(E, F))
        rhs = vadd(br_vf(br_vf(D, E), F), vsmul(s, br_vf(E, br_vf(D, F))))
        if not vzero(vadd(lhs, vsmul(-1, rhs))):
            bad += 1
    return bad


# ================================================= weights and invariant forms

def weight(par, k, lam=Q(0)):
    """the ell_0 = LL_0 eigenvalue on t^{mu+k} theta^S vvol^a."""
    a, mu = par
    return (mu + k[0]) + Q(lam + 1, 1) / 2 * len(k[1]) - a * lam


def weights_checked(par, bas, lam=Q(0)):
    """the same, read off the action of LL_0 (verified diagonal)."""
    L0 = ell(0, lam)
    out = {}
    for ku in bas:
        img = act(par, L0, {ku: Q(1)})
        if set(img) - {ku}:
            raise ValueError("LL_0 is not diagonal on %s: %s" % (ku, img))
        w = img.get(ku, Q(0))
        if w != weight(par, ku, lam):
            raise ValueError("weight formula wrong at %s: %s vs %s"
                             % (ku, w, weight(par, ku, lam)))
        out[ku] = w
    return out


def invariant_forms(par1, par2, gens, N=5, lam=Q(0)):
    """A basis of the invariant bilinear forms V1 x V2 -> C, computed from
    scratch:  B(Xu, v) + (-1)^{p(X)p(u)} B(u, Xv) = 0.

    Only the pairs of total ell_0-weight 0 are unknowns; an equation is kept
    only if every monomial it mentions lies inside the window, so no boundary
    term is silently dropped."""
    bas = keys(range(-N, N + 1))
    w1 = weights_checked(par1, bas, lam)
    w2 = weights_checked(par2, bas, lam)
    idx = {}
    for ku in bas:
        for kv in bas:
            if w1[ku] + w2[kv] == 0:
                idx[(ku, kv)] = len(idx)
    if not idx:
        return [], idx
    inwin = set(bas)
    by_w2 = {}
    for kv in bas:
        by_w2.setdefault(w2[kv], []).append(kv)
    rows = []
    for X in gens:
        pX = vparity(X)
        img1 = {ku: act(par1, X, {ku: Q(1)}) for ku in bas}
        img2 = {kv: act(par2, X, {kv: Q(1)}) for kv in bas}
        wX = None
        for ku in bas:
            for kp in img1[ku]:
                if kp in inwin:
                    wX = w1[kp] - w1[ku]
        if wX is None:
            continue
        for ku in bas:
            for kv in by_w2.get(-wX - w1[ku], ()):
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
                sgn = (-1) ** (pX * modpar(ku))
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
    """(parity, symmetry ratios, non-degeneracy on the inner window)."""
    pars = {(modpar(ku) + modpar(kv)) % 2 for (ku, kv) in form}
    par = pars.pop() if len(pars) == 1 else None
    eps = set()
    for (ku, kv), c in form.items():
        c2 = form.get((kv, ku), Q(0))
        sg = (-1) ** (modpar(ku) * modpar(kv))
        eps.add("degenerate-pair" if c2 == 0 else Q(c) / (sg * c2))
    inner = keys(range(-N + 3, N - 2))
    full = keys(range(-N, N + 1))
    dead = [ku for ku in inner
            if not any(form.get((ku, kv), Q(0)) for kv in full)]
    dead += [kv for kv in inner
             if not any(form.get((ku, kv), Q(0)) for ku in full)]
    return par, eps, (not dead), dead


def check_invariance(form, par1, par2, gens, N):
    """re-verify B(Xu, v) + (-1)^{p(X)p(u)} B(u, Xv) = 0 on the whole window."""
    bas = keys(range(-N, N + 1))
    inwin = set(bas)
    bad = tot_n = 0
    for X in gens:
        pX = vparity(X)
        for ku in bas:
            im1 = act(par1, X, {ku: Q(1)})
            if any(kp not in inwin for kp in im1):
                continue                      # a boundary term would be lost
            for kv in bas:
                im2 = act(par2, X, {kv: Q(1)})
                if any(kp not in inwin for kp in im2):
                    continue
                tot_n += 1
                s = Q(0)
                for kp, co in im1.items():
                    s += co * form.get((kp, kv), Q(0))
                sgn = (-1) ** (pX * modpar(ku))
                for kp, co in im2.items():
                    s += sgn * co * form.get((ku, kp), Q(0))
                if s:
                    bad += 1
    return bad, tot_n


# ================================================= the pairing as a closed rule
#
# The form found by invariant_forms() is known only inside the window in which
# it was solved.  Every one of them turns out to be the residue pairing
#
#       B( (k, S), (k', S') ) = eps(S) delta_{S', S^c} delta_{k + k', K},
#
# with a constant K and a sign eps(S) independent of k; PairRule reads K and
# eps off the computed form, VERIFIES that the form is exactly of this shape on
# the whole window it was solved in, and then supplies the same form on any
# window -- which is what the Clifford-Weyl computations need.

class PairRule:
    def __init__(self, form):
        K, eps = None, {}
        for ((k, S), (k2, S2)), c in form.items():
            if not c:
                continue
            Sc = tuple(i for i in (1, 2) if i not in S)
            if S2 != Sc:
                raise ValueError("the pairing is not S <-> S^c: %s" % (form,))
            if K is None:
                K = k + k2
            elif k + k2 != K:
                raise ValueError("the pairing has no constant t-degree")
            if S in eps and eps[S] != c:
                raise ValueError("the pairing is not k-independent at S=%s" % (S,))
            eps[S] = c
        if K is None:
            raise ValueError("empty form")
        self.K, self.eps = K, eps

    def get(self, key, default=Q(0)):
        (k, S), (k2, S2) = key
        Sc = tuple(i for i in (1, 2) if i not in S)
        if S2 != Sc or k + k2 != self.K:
            return default
        return self.eps.get(S, default)

    def __contains__(self, key):
        return self.get(key, None) is not None

    def items(self):
        raise RuntimeError("PairRule has no finite item list; use .get()")

    def as_dict(self, N):
        out = {}
        for k in keys(range(-N, N + 1)):
            for k2 in keys(range(-N, N + 1)):
                v = self.get((k, k2))
                if v:
                    out[(k, k2)] = v
        return out

    def agrees_with(self, form):
        bad = 0
        for key, c in form.items():
            if self.get(key) != c:
                bad += 1
        return bad


# ============================================================ the doubled space
#
# W = V (+) V^#, or W = V alone when V is self-dual; modes are (s, a, S) with
# s = 0 for V and s = 1 for V^# (s = 0 only in the undoubled case).

class Wspace:
    """The same interface as ns2.Wspace, but the polarization is by the
    ell_0-weight, and the weight-0 modes are dealt with explicitly."""

    def __init__(self, par, par_d, pair, lam=Q(0), par_shift=0,
                 undoubled=False, zero_creation=()):
        self.par, self.par_d, self.pair = par, par_d, pair
        self.lam = lam
        self.par_shift = par_shift
        self.undoubled = undoubled
        self.sides = (0,) if undoubled else (0, 1)
        self._w = {}
        self.zero_creation = set(zero_creation)
        # sanity: sigma is an involution reversing the weight
        for m in self.modes(4):
            sm = self.sigma(m)
            if self.sigma(sm) != m:
                raise ValueError("sigma is not an involution at %s" % (m,))
            if self.w(m) + self.w(sm) != 0:
                raise ValueError("sigma does not reverse the weight at %s: %s %s"
                                 % (m, self.w(m), self.w(sm)))
            if self.beta(m) == 0:
                raise ValueError("the pairing is degenerate at %s" % (m,))

    # ------------------------------------------------------------ bookkeeping
    def w(self, m):
        if m not in self._w:
            s, a, S = m
            p = self.par if s == 0 else self.par_d
            self._w[m] = weight(p, (a, S), self.lam)
        return self._w[m]

    def par_(self, m):
        return (len(m[2]) + (self.par_shift if m[0] == 1 else 0)) % 2

    def sigma(self, m):
        s, a, S = m
        Sc = tuple(i for i in (1, 2) if i not in S)
        return (0 if self.undoubled else 1 - s, self.pair.K - a, Sc)

    def B(self, m1, m2):
        (s1, a1, S1), (s2, a2, S2) = m1, m2
        if self.undoubled:
            #  the form lives on V x V and is given in full
            return self.pair.get(((a1, S1), (a2, S2)), Q(0))
        if s1 == s2:
            return Q(0)
        if s1 == 0:
            return self.pair.get(((a1, S1), (a2, S2)), Q(0))
        return (-1) ** (self.par_(m1) * self.par_(m2)) * \
            self.pair.get(((a2, S2), (a1, S1)), Q(0))

    def beta(self, m):
        return self.B(self.sigma(m), m)

    def act(self, D, m):
        s, a, S = m
        p = self.par if s == 0 else self.par_d
        r = act(p, D, {(a, S): Q(1)})
        return {(s, k[0], k[1]): v for k, v in r.items()}

    def is_creation(self, m):
        w = self.w(m)
        if w < 0:
            return True
        if w > 0:
            return False
        return m in self.zero_creation

    def zero_modes(self, window):
        return [m for m in self.modes(window) if self.w(m) == 0]

    def modes(self, window):
        """the window is centered on the pairing, i.e., it is symmetric under
        k |--> K - k; otherwise the truncation is lopsided and the answers
        depend on it (the module cF_mu = cF_{mu+1} would get two answers)."""
        K = self.pair.K
        lo = -((2 * window - K + 1) // 2)          # ceil(K/2 - window)
        hi = (2 * window + K) // 2                 # floor(K/2 + window)
        return [(s, Q(a), S) for s in self.sides
                for a in range(lo, hi + 1) for S in SUBSETS]


def default_zero_creation(W, window):
    """one member of each weight-0 sigma-couple, chosen by the ordering of
    the keys; the other choices are run separately."""
    out, seen = [], set()
    for m in W.zero_modes(window):
        if m in seen:
            continue
        sm = W.sigma(m)
        seen.add(m)
        seen.add(sm)
        out.append(min(m, sm))
    return out


def zero_couples(W, window):
    out, seen = [], set()
    for m in W.zero_modes(window):
        if m in seen:
            continue
        sm = W.sigma(m)
        seen.add(m)
        seen.add(sm)
        out.append((m, sm))
    return out


# --------------------------------------------------- the Clifford-Weyl algebra
#
# psi, hatX, sadd, tp and VAC are imported from ns2.py; they use only the
# interface above.  The B-relations are re-verified in runvect2.py.

def test_hat(W, window, fields):
    """[hat X, psi_v] = psi_{Xv} (super sense) -- the defining property."""
    bad, tot = 0, 0
    for D in fields:
        pD = vparity(D)
        for m in W.modes(2):
            tot += 1
            sgn = -((-1) ** (pD * tp(W, m)))
            lhs = sadd(hatX(W, D, psi(W, m, VAC), window),
                       psi(W, m, hatX(W, D, VAC, window)), Q(sgn))
            rhs = {}
            for p_, co in W.act(D, m).items():
                rhs = sadd(rhs, psi(W, p_, VAC), co)
            if sadd(lhs, rhs, Q(-1)):
                bad += 1
    return bad, tot


def cw_relations_bad(W, window):
    """psi_m psi_n - (-1)^{tp tp} psi_n psi_m = B(m, n) on the vacuum and on
    one-particle states."""
    bad, tot = 0, 0
    ms = W.modes(window)
    for m, n in product(ms, repeat=2):
        for st in (VAC,) + tuple({(k,): Q(1)} for k in ms[:4]):
            tot += 1
            lhs = sadd(psi(W, m, psi(W, n, st)),
                       psi(W, n, psi(W, m, st)),
                       Q(-((-1) ** (tp(W, m) * tp(W, n)))))
            rhs = {k: W.B(m, n) * v for k, v in st.items() if W.B(m, n)}
            if sadd(lhs, rhs, Q(-1)):
                bad += 1
    return bad, tot


def cocycle(W, D, E, window):
    """the scalar [hat D, hat E] - hat [D, E] on the vacuum (D, E even)."""
    lhs = sadd(hatX(W, D, hatX(W, E, VAC, window), window),
               hatX(W, E, hatX(W, D, VAC, window), window), Q(-1))
    br = br_vf(D, E)
    if not vzero(br):
        lhs = sadd(lhs, hatX(W, br, VAC, window), Q(-1))
    return lhs.get((), Q(0)), {k: v for k, v in lhs.items() if k != ()}


# ---------------------------------------------------------------- the labels

def cubic(W, window, lam=Q(0), ms=(2, 3), checks=(4,)):
    """c(LL_m, LL_{-m}) = alpha m^3 + kappa m: fit on ms, then check the
    prediction at the further values `checks`.  Returns (alpha, kappa, values,
    ok), where ok says whether every check agreed; a check fails when the
    window is too small for LL_{+-m}, which is why it is reported and not
    raised -- the honest test of the fit is the stability in the window."""
    vals = []
    for m in ms:
        v, rest = cocycle(W, ell(m, lam), ell(-m, lam), window)
        assert not rest, "the cocycle is not central at m = %d: %s" % (m, rest)
        vals.append((m, v))
    (m1, v1), (m2, v2) = vals[0], vals[1]
    det = m1 ** 3 * m2 - m1 * m2 ** 3
    al = (v1 * m2 - m1 * v2) / det
    ka = (m1 ** 3 * v2 - v1 * m2 ** 3) / det
    ok = True
    for m in checks:
        v, rest = cocycle(W, ell(m, lam), ell(-m, lam), window)
        vals.append((m, v))
        if rest or al * m ** 3 + ka * m != v:
            ok = False
    return al, ka, vals, ok


def hat_eigen(W, D, window):
    """the eigenvalue of hat D on the vacuum (asserted to be a scalar)."""
    st = hatX(W, D, VAC, window)
    rest = {k: v for k, v in st.items() if k != ()}
    if rest:
        raise ValueError("hat D does not fix the vacuum: %s" % rest)
    return st.get((), Q(0))


def labels(W, window, lam=Q(0), ms=(2, 3)):
    """(c, h, h_from_commutator, kappa, cubic_ok)."""
    al, ka, _, ok = cubic(W, window, lam, ms)
    c = 12 * al
    h0 = hat_eigen(W, ell(0, lam), window)
    st = hatX(W, ell(1, lam), hatX(W, ell(-1, lam), VAC, window), window)
    v11 = st.get((), Q(0))
    #  <ell_1 ell_-1> = <[ell_1, ell_-1]> = -2h + (alpha + kappa)
    hc = (al + ka - v11) / 2
    return c, h0, hc, ka, ok
