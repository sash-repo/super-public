#!/usr/bin/env python3
r"""
mumford.py -- the Mumford-type exponents of the distinguished stringy
superalgebras.

For a module V of tensor fields over a stringy superalgebra g and a Witt
subalgebra {ell_m} of g, the semi-infinite (Japanese) cocycle

    psi(A, B) = sum_{u in V_-, v in V_+} sgn(u) ( B[v,u] A[u,v] - A[v,u] B[u,v] ),

V_- = the modes of negative ell_0-weight, gives psi(ell_m, ell_{-m}) = a m^3 + b m;
c(V) := -12 a  (the sign fixed by c(F_2) = -26 over Witt, the b,c ghosts).
The supertrace sign sgn(u) = (-1)^{p(u)} is the statistics of the semi-infinite
wedge: an even mode gives a fermion.  A change of the polarization by finitely
many modes changes psi by a coboundary (linear in m), so c does not see it; a
finite-dimensional subquotient contributes 0; on a module with a filtration by
submodules adapted to the basis, c is additive.

The Mumford-type exponent:  e(g) = c(g) / c(Ber), the adjoint module and the
Berezinian (volume) module each in its natural parity.  The localization
principle (Beilinson--Schechtman, Kontsevich, Arbarello--De Concini--Kac--Procesi)
turns c(V) into the class of the determinant (Berezinian) line of V.

Everything is exact (Fractions).  Run:  python3 mumford.py  > mumford.log
"""

from fractions import Fraction as Q
from itertools import combinations
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))
from exactlin import nullspace, solve, rank  # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


CHECKS = [0, 0]


def check(cond, msg):
    CHECKS[0] += 1
    if cond:
        CHECKS[1] += 1
    else:
        say("FAIL:", msg)


# ======================================================= Grassmann x Laurent
# a function: {(a, S): coeff}, a = Fraction (power of the even variable x),
# S = sorted tuple of odd indices 1..n.  Monomial x^a th_{S0} th_{S1} ...

def clean(f):
    return {k: v for k, v in f.items() if v != 0}


def add(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, 0) + v
    return clean(out)


def smul(c, f):
    return clean({k: c * v for k, v in f.items()})


def merge(S, T):
    if set(S) & set(T):
        return None, 0
    lst = list(S) + list(T)
    sign = 1
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                sign = -sign
    return tuple(sorted(lst)), sign


def mul(f, g):
    out = {}
    for (a, S), c in f.items():
        for (b, T), d in g.items():
            U, s = merge(S, T)
            if s:
                k = (a + b, U)
                out[k] = out.get(k, 0) + s * c * d
    return clean(out)


def mono(a, S=(), c=1):
    return {(Q(a), tuple(sorted(S))): Q(c)}


def parity(f):
    ps = {len(S) % 2 for (a, S) in f}
    assert len(ps) <= 1, f
    return ps.pop() if ps else 0


def d_x(f):
    return clean({(a - 1, S): a * c for (a, S), c in f.items()})


def d_th(i, f):
    """left derivative."""
    out = {}
    for (a, S), c in f.items():
        if i in S:
            k = S.index(i)
            T = S[:k] + S[k + 1:]
            out[(a, T)] = out.get((a, T), 0) + (-1) ** k * c
    return clean(out)


# ============================================================ vector fields
# D = {"x": f_x, i: f_i}, coefficients on the left; p(D) = p(f_x).

def vf(n, **kw):
    D = {"x": {}}
    for i in range(1, n + 1):
        D[i] = {}
    for k, v in kw.items():
        D["x" if k == "x" else int(k[1:])] = clean(v)
    return D


def vadd(*Ds):
    out = {}
    for D in Ds:
        for k, v in D.items():
            out[k] = add(out.get(k, {}), v)
    return out


def vsmul(c, D):
    return {k: smul(c, v) for k, v in D.items()}


def vpar(D):
    ps = set()
    for k, v in D.items():
        if v:
            ps.add(parity(v) if k == "x" else (parity(v) + 1) % 2)
    assert len(ps) <= 1, D
    return ps.pop() if ps else 0


def apply(D, u):
    out = mul(D["x"], d_x(u))
    for k, v in D.items():
        if k != "x" and v:
            out = add(out, mul(v, d_th(k, u)))
    return out


def br(D, E):
    s = (-1) ** (vpar(D) * vpar(E))
    return {k: add(apply(D, E[k]), smul(-s, apply(E, D[k]))) for k in D}


def Div(D):
    out = d_x(D["x"])
    for k, v in D.items():
        if k != "x" and v:
            out = add(out, smul((-1) ** parity(v), d_th(k, v)))
    return out


def vzero(D):
    return not any(v for v in D.values())


# ============================================================== modules
# A module is a function act(D, u) -> u' on "elements" u (dicts of any keys),
# a parity function on basis keys, and a basis generator.

def subsets(n):
    out = []
    for r in range(n + 1):
        out += list(combinations(range(1, n + 1), r))
    return out


class Module:
    """basis = list of (key, element); element is a dict over 'coords'.
    coords(u) -> dict {coord: coeff}; basis elements are ell_0-eigenvectors."""

    def __init__(self, name, basis, act, par, coords):
        self.name, self.basis, self.act, self.par, self.coords = \
            name, basis, act, par, coords


def energies(M, ell0):
    """ell_0-eigenvalue of each basis vector (checked to be an eigenvector)."""
    E = {}
    for key, u in M.basis:
        w = M.act(ell0, u)
        cu, cw = M.coords(u), M.coords(w)
        piv = next(iter(cu))
        lam = cw.get(piv, 0) / cu[piv]
        check(clean(add(cw, smul(-lam, cu))) == {},
              "%s: basis vector %s is not an ell_0-eigenvector" % (M.name, key))
        E[key] = lam
    return E


def matrix_in_basis(M, D, keys):
    """columns: D acting on basis vectors with key in keys, expressed in the
    whole basis via coordinates (basis must be coordinate-triangular: we solve)."""
    raise NotImplementedError


def psi(M, ellm, ellmm, E, coord_of):
    """the Japanese cocycle psi(ell_m, ell_{-m}) on M.
    coord_of(u) -> {key: coeff} expresses u in the basis of M."""
    A, B = {}, {}
    for key, u in M.basis:
        A[key] = coord_of(M.act(ellm, u))
        B[key] = coord_of(M.act(ellmm, u))
    neg = [k for k, _ in M.basis if E[k] < 0]
    pos = set(k for k, _ in M.basis if E[k] >= 0)
    s = Q(0)
    for u in neg:
        sg = (-1) ** M.par(u)
        for v, Avu in A[u].items():       # A[v,u]: A applied to u, coefficient of v
            if v in pos:
                s -= sg * Avu * B[v].get(u, 0)
        for v, Bvu in B[u].items():
            if v in pos:
                s += sg * Bvu * A[v].get(u, 0)
    return s


def charge(build, ell, ms=(1, 2, 3, 4), name=""):
    """build(m) -> (Module, coord_of) on a window wide enough for ell_{+-m};
    returns c with the fit verified on the 4th value of m."""
    vals = {}
    for m in ms:
        M, coord_of = build(m)
        E = energies(M, ell(0))
        vals[m] = psi(M, ell(m), ell(-m), E, coord_of)
    # psi = a m^3 + b m
    a = (vals[2] - 2 * vals[1]) / 6
    b = vals[1] - a
    for m in ms:
        check(vals[m] == a * m ** 3 + b * m,
              "%s: psi(%d) = %s is not a m^3 + b m" % (name, m, vals[m]))
    return -12 * a


# ===================================================== the concrete modules

def window_monos(n, W, shift=lambda S: Q(0), xw=Q(1), thw=None):
    """monomials x^(k+shift(S)) th^S with |energy| <= W, energy estimated as
    xw*k + sum th weights; we take a generous range of k."""
    out = []
    for S in subsets(n):
        for k in range(-2 * W - 4, 2 * W + 5):
            out.append((Q(k) + shift(S), S))
    return out


def density_module(n, ell, gen_par, mult, w, W, shift=lambda S: Q(0),
                   name="dens"):
    """u = f * gen, D.u = (D(f) + w*mult(D)*f) gen; basis = monomials with
    |ell_0-weight| <= W."""
    def act(D, u):
        return add(apply(D, u), smul(w, mul(mult(D), u)))
    basis = []
    for (a, S) in window_monos(n, W, shift):
        u = {(a, S): Q(1)}
        basis.append(((a, S), u))
    M = Module(name, basis, act,
               lambda key: (len(key[1]) + gen_par) % 2, lambda u: u)
    E = energies(M, ell(0))
    keep = [(k, u) for k, u in basis if abs(E[k]) <= W]
    keys = set(k for k, _ in keep)
    M = Module(name, keep, act, M.par, lambda u: u)
    return M, (lambda u: {k: c for k, c in u.items() if k in keys})


# ------------------------------------------------------------ contact

def K(f, n):
    """K_f = (2-E)(f) d_x - (-1)^{p(f)} sum_i (d_i f - th_i d_x f) d_i."""
    Ef = {}
    for (a, S), c in f.items():
        Ef[(a, S)] = (2 - len(S)) * c
    D = vf(n, x=clean(Ef))
    s = (-1) ** parity(f)
    for i in range(1, n + 1):
        D[i] = smul(-s, add(d_th(i, f), smul(-1, mul(mono(0, (i,)), d_x(f)))))
    return D


def contact_ell(n):
    return lambda m: vsmul(Q(1, 2), K(mono(m + 1), n))


def contact_mult(n):
    # L_{K_f} alpha = 2 d_x(f) alpha; recover f from K_f: f = alpha(K_f) =
    # f_x + sum th_i f_i  (alpha = dx + sum th_i dth_i)
    def mult(D):
        f = D["x"]
        for i in range(1, n + 1):
            f = add(f, mul(mono(0, (i,)), D[i]))
        return d_x(f)          # = 2 d_x(g) for D = K_g, since alpha(K_g) = 2g
    return mult


def ramond_shift(n):
    return lambda S: Q(1, 2) if n in S else Q(0)


# ------------------------------------------------------------ vect, svect

def vect_ell(n, w=Q(1, 2), kappa=Q(0)):
    def ell(m):
        D = vf(n, x=mono(m + 1))
        for i in range(1, n + 1):
            D[i] = mono(m, (i,), w * (m + 1) + kappa)
        return D
    return ell


def field_basis(n, W, ell, sub=None, name="adj"):
    """adjoint module of vect(1|n) or of its subalgebra sub (a function giving,
    for a list of fields of one weight, a basis of the subalgebra there)."""
    raw = []
    for S in subsets(n):
        for k in range(-2 * W - 4, 2 * W + 5):
            for var in ["x"] + list(range(1, n + 1)):
                D = vf(n)
                D[var] = mono(k, S)
                raw.append(D)
    l0 = ell(0)
    byE = {}
    for D in raw:
        Dl = br(l0, D)
        # eigen check on the monomial field
        lam = None
        for var, f in D.items():
            if f:
                (key, c), = f.items()
                lam = Dl[var].get(key, 0) / c
        check(vzero(vadd(Dl, vsmul(-lam, D))), "monomial field not ell0-eigen")
        if abs(lam) <= W:
            byE.setdefault(lam, []).append(D)
    basis = []
    for lam, Ds in byE.items():
        if sub is not None:
            Ds = sub(Ds)
        for j, D in enumerate(Ds):
            basis.append(((lam, j), D))
    return basis


def flat(D):
    out = {}
    for var, f in D.items():
        for key, c in f.items():
            out[(var, key)] = c
    return out


def field_module(n, W, ell, sub=None, name="adj"):
    basis = field_basis(n, W, ell, sub, name)
    byE = {}
    for key, D in basis:
        byE.setdefault(key[0], []).append((key, D))
    # coordinates: solve within the weight space
    cache = {}

    def coord_of(D):
        if vzero(D):
            return {}
        # weight of D
        lam = None
        l0 = ell(0)
        Dl = br(l0, D)
        fD = flat(D)
        piv = next(iter(fD))
        lam = flat(Dl).get(piv, 0) / fD[piv]
        if lam not in byE:
            return {}            # outside the window
        vecs = byE[lam]
        cols = sorted(set().union(*[flat(V).keys() for _, V in vecs])
                      | set(fD.keys()), key=str)
        Mt = [[flat(V).get(c, 0) for _, V in vecs] for c in cols]
        rhs = [fD.get(c, 0) for c in cols]
        x = solve(Mt, rhs)
        check(x is not None, "%s: element not in the subalgebra" % name)
        return {vecs[j][0]: x[j] for j in range(len(vecs)) if x[j] != 0}

    M = Module(name, basis, lambda D, X: br(D, X), None, flat)
    pars = {key: vpar(D) for key, D in basis}
    M.par = lambda key: pars[key]
    return M, coord_of


def svect_sub(n, lam):
    def sub(Ds):
        # kernel of div_lam(D) = Div D + lam * f_x / x
        cols = {}
        imgs = []
        for D in Ds:
            g = add(Div(D), smul(lam, mul(mono(-1), D["x"])))
            imgs.append(g)
        keys = sorted(set().union(*[g.keys() for g in imgs]) or {None}, key=str)
        if keys == [None]:
            return Ds
        Mt = [[g.get(k, 0) for g in imgs] for k in keys]
        ker = nullspace(Mt)
        out = []
        for v in ker:
            out.append(vadd(*[vsmul(v[j], Ds[j]) for j in range(len(Ds))
                              if v[j] != 0]))
        return out
    return sub


def svect_ell(lam):
    def ell(m):
        D = vf(2, x=mono(m + 1))
        half = (lam + m + 1) / Q(2)
        D[1] = mono(m, (1,), half)
        D[2] = mono(m, (2,), half)
        return D
    return ell


# ------------------------------------------------------------ m^L(1)
# variables: x = q (even), odd 1 = tau, 2 = xi.
# M_f = (2-E)(f) d_tau - (d_q f d_xi + (-1)^{p(f)} d_xi f d_q)
#       - (-1)^{p(f)} d_tau f E,   E = xi d_xi + q d_q.

def Mf(f):
    n = 2
    Ef = {}
    for (a, S), c in f.items():
        Ef[(a, S)] = (2 - (a + (1 if 2 in S else 0))) * c
    s = (-1) ** parity(f)
    D = vf(n)
    D[1] = clean(Ef)
    D[2] = smul(-1, d_x(f))
    D["x"] = smul(-s, d_th(2, f))
    tf = d_th(1, f)
    D[2] = add(D[2], smul(-s, mul(tf, mono(0, (2,)))))
    D["x"] = add(D["x"], smul(-s, mul(tf, mono(1))))
    return D


def m_ell(m):
    """[M_{q^a tau}, M_{q^b tau}] = (b - a) M_{q^{a+b} tau}, verified below."""
    return Mf(mono(m, (1,)))


def m_module(W, ell=m_ell):
    """the adjoint module of m^L(1): basis M_f, f = q^k tau^e xi^d."""
    raw = []
    for S in subsets(2):
        for k in range(-2 * W - 4, 2 * W + 5):
            raw.append(Mf(mono(k, S)))
    l0 = ell(0)
    byE = {}
    for D in raw:
        if vzero(D):
            continue
        Dl = br(l0, D)
        fD = flat(D)
        piv = next(iter(fD))
        lam = flat(Dl).get(piv, 0) / fD[piv]
        check(vzero(vadd(Dl, vsmul(-lam, D))), "M_f not ell0-eigen")
        if abs(lam) <= W:
            byE.setdefault(lam, []).append(D)
    basis = []
    for lam, Ds in byE.items():
        check(rank([[flat(D).get(c, 0) for D in Ds] for c in
                    sorted(set().union(*[flat(D).keys() for D in Ds]), key=str)])
              == len(Ds), "M injective")
        for j, D in enumerate(Ds):
            basis.append(((lam, j), D))
    pars = {key: vpar(D) for key, D in basis}

    def coord_of(D):
        if vzero(D):
            return {}
        fD = flat(D)
        piv = next(iter(fD))
        lam = flat(br(ell(0), D)).get(piv, 0) / fD[piv]
        if lam not in byE:
            return {}
        vecs = [(k, V) for k, V in basis if k[0] == lam]
        cols = sorted(set().union(*[flat(V).keys() for _, V in vecs])
                      | set(fD.keys()), key=str)
        x = solve([[flat(V).get(c, 0) for _, V in vecs] for c in cols],
                  [fD.get(c, 0) for c in cols])
        check(x is not None, "m: bracket left the subalgebra")
        return {vecs[j][0]: x[j] for j in range(len(vecs)) if x[j] != 0}
    M = Module("m adj", basis, lambda D, X: br(D, X), None, flat)
    M.par = lambda key: pars[key]
    return M, coord_of


# ================================================================== main

def main():
    say("mumford.py -- Mumford-type exponents of the distinguished stringy "
        "superalgebras")
    W = 6

    # ---------------- Witt (n = 0): F_lam = f (dx)^lam, ell_m = x^{m+1} d_x
    def witt_ell(m):
        return vf(0, x=mono(m + 1))

    def witt_F(lam):
        return lambda m: density_module(0, witt_ell, 0, lambda D: d_x(D["x"]),
                                        lam, W + m, name="F%s" % lam)
    say("\n[calibration] Witt, c(F_lam) = -2(6 lam^2 - 6 lam + 1):")
    for lam in [Q(2), Q(1), Q(0), Q(1, 2), Q(-1), Q(3, 2), Q(1, 3)]:
        c = charge(witt_F(lam), witt_ell, name="witt F%s" % lam)
        say("  lam = %-5s c = %s" % (lam, c))
        check(c == -2 * (6 * lam ** 2 - 6 * lam + 1), "witt F%s" % lam)
    # the adjoint of Witt as fields:
    cadj = charge(lambda m: field_module(0, W + m, witt_ell), witt_ell,
                  name="witt adj")
    say("  adjoint (fields):", cadj)
    check(cadj == -26, "witt adjoint")

    # ---------------- contact k^L(1|n), k^M(1|n)
    res = {}
    for n in range(1, 5):
        for sector in ("L", "M"):
            shift = (lambda S: Q(0)) if sector == "L" else ramond_shift(n)
            ell = contact_ell(n)
            mult = contact_mult(n)

            def F(lam, gpar):
                return lambda m: density_module(
                    n, ell, gpar, mult, lam / 2, W + m, shift,
                    name="k%s(1|%d) F%s" % (sector, n, lam))
            # family check: c(F_lam) with even generator
            if n <= 2:
                for lam in [Q(-2), Q(0), Q(1), Q(3), Q(1, 2)]:
                    c = charge(F(lam, 0), ell, name="k F")
                    exp = {1: 6 * lam - 3, 2: Q(-6)}[n]
                    check(c == exp, "k%s(1|%d): c(F%s) = %s, expected %s"
                          % (sector, n, lam, c, exp))
            # adjoint: K_f, f in F_{-2}; cross-check against the bracket
            cadj = charge(F(Q(-2), 0), ell, name="adj")
            ber = Q(2 - n)
            cber = charge(F(ber, n % 2), ell, name="ber")
            res[("k" + sector, n)] = (cadj, cber)
            say("k^%s(1|%d): c(adj = F_-2) = %s, c(Ber = F_%s, parity %d) = %s"
                % (sector, n, cadj, ber, n % 2, cber))
    # bracket = module action on F_{-2}, i.e., [K_g, K_f] = K_{g.f}:
    for n in (1, 2, 3):
        for f, g in [(mono(2, (1,)), mono(-1)), (mono(1, (1, 2) if n > 1 else (1,)),
                                                 mono(0, (1,)))]:
            lhs = br(K(g, n), K(f, n))
            u = add(apply(K(g, n), f), smul(-1, mul(contact_mult(n)(K(g, n)), f)))
            check(vzero(vadd(lhs, vsmul(-1, K(u, n)))),
                  "adjoint of k(1|%d) is F_-2" % n)

    # ---------------- vect(1|1), vect(1|2) with several Witt copies
    for n in (1, 2):
        for w, kappa in [(Q(1, 2), Q(0)), (Q(1, 3), Q(0)), (Q(1, 4), Q(1, 5))]:
            ell = vect_ell(n, w, kappa)
            for a, b in [(1, 3), (2, 5)]:
                check(vzero(vadd(br(ell(a), ell(b)), vsmul(-(b - a), ell(a + b)))),
                      "vect Witt copy closes")
            cadj = charge(lambda m: field_module(n, W + m, ell), ell,
                          name="vect adj")
            cber = charge(lambda m: density_module(n, ell, n % 2, Div, Q(1),
                                                   W + m), ell, name="vect ber")
            cfun = charge(lambda m: density_module(n, ell, 0, Div, Q(0),
                                                   W + m), ell, name="vect fun")
            say("vect(1|%d), w = %s, kappa = %s: c(adj) = %s, c(Ber) = %s, "
                "c(functions) = %s, e = %s" % (n, w, kappa, cadj, cber, cfun,
                                               cadj / cber))
            res[("vect", n, w, kappa)] = (cadj, cber)

    # ---------------- svect_lam(1|2)
    for lam in [Q(1, 3), Q(1, 2), Q(2, 7)]:
        ell = svect_ell(lam)
        check(add(Div(ell(3)), smul(lam, mul(mono(-1), ell(3)["x"]))) == {},
              "ell in svect")
        cadj = charge(lambda m: field_module(2, W + m, ell, svect_sub(2, lam),
                                             name="svect adj"), ell,
                      name="svect adj")
        cber = charge(lambda m: density_module(2, ell, 0, Div, Q(1), W + m),
                      ell, name="svect ber")
        say("svect_%s(1|2): c(adj) = %s, c(Vol) = %s, e = %s"
            % (lam, cadj, cber, cadj / cber))
        res[("svect", lam)] = (cadj, cber)

    # ---------------- m^L(1)
    for a, b in [(0, 1), (-1, 2), (2, 3), (-3, 1)]:
        check(vzero(vadd(br(m_ell(a), m_ell(b)), vsmul(-(b - a), m_ell(a + b)))),
              "m Witt copy closes")
    cadj = charge(lambda m: m_module(W + m), m_ell, name="m adj")
    cber = charge(lambda m: density_module(2, m_ell, 0, Div, Q(1), W + m),
                  m_ell, name="m ber")
    cfun = charge(lambda m: density_module(2, m_ell, 0, Div, Q(0), W + m),
                  m_ell, name="m fun")
    say("m^L(1): c(adj) = %s, c(Ber) = %s, c(functions) = %s"
        % (cadj, cber, cfun))
    for a in [Q(2), Q(-1), Q(1, 2), Q(3)]:
        say("   m^L(1): c(Vol^%s) = %s" % (a, charge(
            lambda m: density_module(2, m_ell, 0, Div, a, W + m), m_ell)))
    # Vol^a over vect(1|1): the whole family
    ell = vect_ell(1)
    for a in [Q(0), Q(2), Q(3), Q(-1), Q(1, 2)]:
        say("   vect(1|1): c(Vol^%s, even generator) = %s" % (a, charge(
            lambda m: density_module(1, ell, 0, Div, a, W + m), ell)))

    say("\nchecks: %d of %d passed" % (CHECKS[1], CHECKS[0]))
    with open("mumford.log", "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()


# ============================================ the class in H^2, without a probe
# For an algebra whose Witt copies may all miss the cocycle (m^L(1)), compare
# the cocycles psi_V on all pairs (X, Y) of opposite weights directly:
# solve psi_adj - e psi_Ber = delta(phi), phi a functional on the weight-0 part.

def psi_pair(M, coord_of, E, X, Y):
    pX, pY = vpar(X), vpar(Y)
    A, B = {}, {}
    for key, u in M.basis:
        A[key] = coord_of(M.act(X, u))
        B[key] = coord_of(M.act(Y, u))
    neg = [k for k, _ in M.basis if E[k] < 0]
    pos = set(k for k, _ in M.basis if E[k] >= 0)
    s = Q(0)
    sg2 = (-1) ** (pX * pY)
    for u in neg:
        sg = (-1) ** M.par(u)
        for v, Bvu in B[u].items():          # B[v,u]
            if v in pos:
                s += sg * A[v].get(u, 0) * Bvu       # A[u,v] B[v,u]
        for v, Avu in A[u].items():          # A[v,u]
            if v in pos:
                s -= sg2 * sg * B[v].get(u, 0) * Avu
    return s


def h2_ratio(gbasis, coord_g, ell0, mods, wmax, name):
    """gbasis: list of (key, field) with key[0] = weight; coord_g: field ->
    coords in gbasis; mods: {label: (Module, coord_of)} built on a window that
    covers weights up to wmax.  Returns {label: e} with psi_label = e psi_ref +
    coboundary, ref = the first label; and whether psi_ref is a coboundary."""
    byw = {}
    for key, X in gbasis:
        byw.setdefault(key[0], []).append((key, X))
    zero = [k for k, _ in byw.get(Q(0), [])]
    pairs = []
    for w in sorted(byw):
        if 0 < w <= wmax and -w in byw:
            for kx, X in byw[w]:
                for ky, Y in byw[-w]:
                    pairs.append((X, Y))
    Es = {lab: energies(M, ell0) for lab, (M, co) in mods.items()}
    vals = {lab: [psi_pair(M, co, Es[lab], X, Y) for X, Y in pairs]
            for lab, (M, co) in mods.items()}
    brs = [coord_g(br(X, Y)) for X, Y in pairs]
    labs = list(mods)
    ref = labs[0]
    out = {}
    # is psi_ref a coboundary?
    Mt = [[b.get(z, 0) for z in zero] for b in brs]
    x = solve(Mt, vals[ref]) if zero else None
    out["ref_is_coboundary"] = x is not None and all(
        sum(Mt[i][j] * x[j] for j in range(len(zero))) == vals[ref][i]
        for i in range(len(pairs)))
    for lab in labs[1:]:
        Mt2 = [[vals[ref][i]] + [brs[i].get(z, 0) for z in zero]
               for i in range(len(pairs))]
        x = solve(Mt2, vals[lab])
        out[lab] = None if x is None else x[0]
    out["pairs"] = len(pairs)
    return out
