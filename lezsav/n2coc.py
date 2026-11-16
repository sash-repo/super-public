#!/usr/bin/env python3
"""
n2coc.py -- the value of the N=2 classical anomaly.

n2.py produced, in the free supercommutative algebra and with no coordinates,
the super stress tensor of the N=2 super Liouville system (IK2) and its law

    delta_f T = K_f(T) - (lambda + lambdabar) T + A(f),     A(f) = - d D1 D2 f,

where D1 = D_{theta_1}, D2 = D_{theta_2} are the real frame of eq. (Dtheta) and
T := Psi^{-1} D1 D2 Psi with Psi := exp(u), u the dilaton superfield.

Here that anomaly is turned into a number.  We realize k^L(1|2) by its
generating functions on C^{1|2},

    K_f = 2 f d_t - (-1)^{p(f)} ( (D1 f) D1 + (D2 f) D2 ),

verify that this is the contact algebra of the paper (eq. multN), find the
basis L_m, J_m, G_r, Gbar_r in which the structure relations are the standard
N=2 superconformal ones, and evaluate the cocycle

    c(f, g) := Res( g . A(f) ),      Res = the coefficient of t^{-1} th1 th2

on it.  The central charge is then read off three different sectors -- the
Virasoro one, the u(1)-current one and the supercharge one -- which must agree.

Exact arithmetic over the Gaussian rationals throughout (i enters only through
the complex frame Dd = D1 - i D2 of eq. (Dcx), i.e. only through G_r).

Usage:  ../venv/bin/python n2coc.py     (needs sympy)
"""

import itertools
import sympy as sp

# ---------------------------------------------------------------------------
# C[t,t^{-1}] (x) Lambda(theta_1, theta_2):  monomials are (n, s), s a tuple
# ---------------------------------------------------------------------------

TH = (1, 2)


def par(s):
    return len(s) % 2


def th_mul(s1, s2):
    """product of two theta-monomials -> (sign, s) or None."""
    if set(s1) & set(s2):
        return None
    sign = 1
    for a in reversed(s1):
        for b in s2:
            if b < a:
                sign = -sign
    return sign, tuple(sorted(s1 + s2))


class F:
    """an element: dict {(n, s): coefficient}."""

    __slots__ = ("d",)

    def __init__(self, d=None):
        self.d = {} if d is None else {k: sp.expand(v) for k, v in d.items()
                                       if sp.expand(v) != 0}

    @staticmethod
    def mono(n, s=(), c=1):
        return F({(n, tuple(s)): sp.sympify(c)})

    def __add__(self, o):
        d = dict(self.d)
        for k, v in o.d.items():
            d[k] = sp.expand(d.get(k, 0) + v)
            if d[k] == 0:
                del d[k]
        return F(d)

    def __sub__(self, o):
        return self + o.scal(-1)

    def scal(self, c):
        c = sp.sympify(c)
        if c == 0:
            return F()
        return F({k: v * c for k, v in self.d.items()})

    def __mul__(self, o):
        d = {}
        for (n1, s1), c1 in self.d.items():
            for (n2, s2), c2 in o.d.items():
                r = th_mul(s1, s2)
                if r is None:
                    continue
                sg, s = r
                k = (n1 + n2, s)
                d[k] = sp.expand(d.get(k, 0) + sg * c1 * c2)
                if d[k] == 0:
                    del d[k]
        return F(d)

    def is_zero(self):
        return not self.d

    def parity(self):
        ps = {par(s) for (_, s) in self.d}
        assert len(ps) <= 1, "not homogeneous"
        return ps.pop() if ps else 0

    def __repr__(self):
        if not self.d:
            return "0"
        out = []
        for (n, s) in sorted(self.d, key=lambda k: (len(k[1]), k[0], k[1])):
            c = self.d[(n, s)]
            t = f"t^{n}" if n else ""
            th = "".join(f"th{i}" for i in s)
            out.append(f"({c})" + (("*" + t) if t else "") + (("*" + th) if th else ""))
        return " + ".join(out)


ZERO = F()


def dt(e):
    d = {}
    for (n, s), c in e.d.items():
        if n == 0:
            continue
        d[(n - 1, s)] = d.get((n - 1, s), 0) + n * c
    return F(d)


def dth(i, e):
    """left derivative d/d theta_i."""
    d = {}
    for (n, s), c in e.d.items():
        if i not in s:
            continue
        j = s.index(i)
        sg = (-1) ** j
        s2 = s[:j] + s[j + 1:]
        d[(n, s2)] = d.get((n, s2), 0) + sg * c
    return F(d)


def D(i, e):
    """D_{theta_i} = d/d theta_i - theta_i d_t."""
    return dth(i, e) - F.mono(0, (i,)) * dt(e)


def D1(e):
    return D(1, e)


def D2(e):
    return D(2, e)


def K(f, e):
    """K_f(e) = 2 f d_t e - (-1)^{p(f)} ( (D1 f)(D1 e) + (D2 f)(D2 e) )."""
    s = (-1) ** f.parity()
    return f.scal(2) * dt(e) - (D1(f) * D1(e) + D2(f) * D2(e)).scal(s)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def basis(lo=-3, hi=4):
    out = []
    for n in range(lo, hi + 1):
        for s in ((), (1,), (2,), (1, 2)):
            out.append(F.mono(n, s))
    return out


def check_multN():
    """[K_f, D_i] = sum_j (D_i D_j f) D_j  -- eq. (multN) of the paper."""
    n = 0
    for f in basis(-2, 3):
        pf = f.parity()
        for i in (1, 2):
            for e in basis(-2, 3):
                lhs = K(f, D(i, e)) - D(i, K(f, e)).scal((-1) ** pf)
                rhs = ZERO
                for j in (1, 2):
                    rhs = rhs + D(i, D(j, f)) * D(j, e)
                assert (lhs - rhs).is_zero(), (f, i, e, lhs - rhs)
                n += 1
    return n


def gen_of(X, ph):
    """recover h of parity ph from the operator X = K_h.

    K_h(t) = 2h - deg_theta(h) . h, so the theta-degree 0 and 1 parts of h are
    read off K_h(t) (with the factors 2 and 1), while its theta-degree 2 part is
    invisible there and is read off K_h(theta_1) = -(-1)^{p(h)} D_1 h instead."""
    Xt = X(F.mono(1))
    d = {}
    for (n, s), c in Xt.d.items():
        k = len(s)
        if k == 2:
            assert c == 0, "K_h(t) must have no theta_1theta_2 part"
            continue
        d[(n, s)] = c / (2 - k)
    h = F(d)
    if ph == 0:
        # the remaining theta-degree 2 part
        rest = X(F.mono(0, (1,))) - K(h, F.mono(0, (1,))) if not h.is_zero() \
            else X(F.mono(0, (1,)))
        d2 = {}
        for (n, s), c in rest.d.items():
            assert s == (2,), ("unexpected shape", rest)
            d2[(n, (1, 2))] = -c
        h = h + F(d2)
    return h


_BRCACHE = {}
_BRVERIFIED = [0]


def _key(f):
    return tuple(sorted((k, str(v)) for k, v in f.d.items()))


def bracket(f, g):
    """the generating function of [K_f, K_g], verified on a spanning set."""
    ck = (_key(f), _key(g))
    if ck in _BRCACHE:
        return _BRCACHE[ck]
    h = _bracket(f, g)
    _BRCACHE[ck] = h
    return h


def _bracket(f, g):
    pf, pg = f.parity(), g.parity()
    sg = (-1) ** (pf * pg)

    def X(e):
        return K(f, K(g, e)) - K(g, K(f, e)).scal(sg)

    h = gen_of(X, (pf + pg) % 2)
    for e in basis(-1, 2):
        assert (X(e) - K(h, e)).is_zero(), (f, g, e, X(e) - K(h, e))
    _BRVERIFIED[0] += 1
    return h


def res(e):
    """the Berezin residue: the coefficient of t^{-1} th1 th2."""
    return e.d.get((-1, (1, 2)), sp.Integer(0))


def A(f):
    """the anomaly:  A(f) = - d_t D1 D2 f = d_t B(f), B the o(2)-multiplier."""
    return dt(D1(D2(f))).scal(-1)


def rho_action(f, r):
    """the weight-1 density module: delta_f rho = K_f(rho) - (lam+lambar) rho
                                              = K_f(rho) + 2 (d f) rho."""
    return K(f, r) + (dt(f) * r).scal(2)


SIGN = "p(f)"          # the twist in the identification of densities with g^*


def coc(f, g):
    e = {"1": 1,
         "p(f)": (-1) ** f.parity(),
         "p(g)": (-1) ** g.parity(),
         "p(f)p(g)": (-1) ** (f.parity() * g.parity())}[SIGN]
    return res(g * A(f)) * e


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    I = sp.I
    print("=" * 74)
    print("k^L(1|2) as contact fields, and the N=2 anomaly as a number")
    print("=" * 74)
    n = check_multN()
    print(f"   eq. (multN): [K_f,D_i] = sum_j (D_iD_jf) D_j   {n} checks   OK")

    # the contact bracket closes and is super-skew
    n = 0
    for f in basis(-1, 2):
        for g in basis(-1, 2):
            h1 = bracket(f, g)
            h2 = bracket(g, f)
            assert (h1 + h2.scal((-1) ** (f.parity() * g.parity()))).is_zero()
            n += 1
    print(f"   the bracket closes and is super-skew           {n} checks   OK")

    # ---------------- the standard N=2 basis ------------------------------
    cL, cJ, cG, cB = sp.symbols("cL cJ cG cB")
    th = F.mono(0, (1,)) + F.mono(0, (2,)).scal(I)          # theta   = th1 + i th2
    tb = F.mono(0, (1,)) - F.mono(0, (2,)).scal(I)          # thetabar= th1 - i th2

    def L(m, c=cL):
        return F.mono(m + 1).scal(c)

    def J(m, c=cJ):
        return F.mono(m, (1, 2)).scal(c)

    def G(r, c=cG):
        return F.mono(sp.Rational(2 * r + 1, 2)).scal(c) * th   # t^{r+1/2} theta

    def Gb(r, c=cB):
        return F.mono(sp.Rational(2 * r + 1, 2)).scal(c) * tb

    # t^{r+1/2} with r half-integer is an integer power; keep r half-integral
    def Ghalf(r, c=cG):
        e = sp.Rational(2 * r + 1, 2)
        assert e == int(e)
        return F.mono(int(e)).scal(c) * th

    def Gbhalf(r, c=cB):
        e = sp.Rational(2 * r + 1, 2)
        assert e == int(e)
        return F.mono(int(e)).scal(c) * tb

    # -- solve for the normalizations --------------------------------------
    # [L_m, L_n] = (m-n) L_{m+n}
    m, nn = 2, -1
    lhs = bracket(L(m), L(nn))
    tgt = L(m + nn).scal(m - nn)
    eqs = []
    for k in set(lhs.d) | set(tgt.d):
        eqs.append(sp.expand(lhs.d.get(k, 0) - tgt.d.get(k, 0)))
    solL = sp.solve([e for e in eqs if e != 0], cL, dict=True)
    cLval = [s[cL] for s in solL if s[cL] != 0][0]
    print(f"   L_m := K_f with f = ({cLval}) t^(m+1)")

    # {G_r, Gbar_s} = 2 L_{r+s} + (r-s) J_{r+s}   (plus the central term)
    r, s = sp.Rational(3, 2), sp.Rational(-1, 2)
    lhs = bracket(Ghalf(r), Gbhalf(s))
    tgt = L(int(r + s), cLval).scal(2) + J(int(r + s)).scal(r - s)
    eqs = []
    for k in set(lhs.d) | set(tgt.d):
        eqs.append(sp.expand(lhs.d.get(k, 0) - tgt.d.get(k, 0)))
    sol = sp.solve([e for e in eqs if e != 0], [cJ, cG, cB], dict=True)
    print("   {G,Gbar} normalization:", sol)
    # pick the branch with cG = 1 (the residual scaling G -> sG, Gbar -> Gbar/s)
    sub = None
    for ss in sol:
        cand = {k: sp.simplify(v.subs({cG: 1})) if hasattr(v, "subs") else v
                for k, v in ss.items()}
        cand[cG] = sp.Integer(1)
        if cand.get(cB, 0) != 0 and cand.get(cJ, 0) != 0:
            sub = cand
            break
    assert sub is not None, sol
    cJval, cGval, cBval = sub[cJ], sub[cG], sub[cB]
    print(f"   J_m := K_f with f = ({cJval}) t^m th1 th2")
    print(f"   G_r := K_f with f = ({cGval}) t^(r+1/2) (th1 + i th2)")
    print(f"   Gbar_r := K_f with f = ({cBval}) t^(r+1/2) (th1 - i th2)")

    LL = lambda m: L(m, cLval)
    JJ = lambda m: J(m, cJval)
    GG = lambda r: Ghalf(r, cGval)
    BB = lambda r: Gbhalf(r, cBval)

    # -- verify the whole set of structure relations -----------------------
    print()
    print("   verifying the N=2 superconformal relations:")
    nchk = 0
    R = range(-3, 4)
    HR = [sp.Rational(2 * k + 1, 2) for k in range(-3, 3)]
    for m in R:
        for n_ in R:
            assert (bracket(LL(m), LL(n_)) - LL(m + n_).scal(m - n_)).is_zero()
            assert (bracket(LL(m), JJ(n_)) + JJ(m + n_).scal(n_)).is_zero()
            assert bracket(JJ(m), JJ(n_)).is_zero()
            nchk += 3
    for m in R:
        for r in HR:
            assert (bracket(LL(m), GG(r)) - GG(m + r).scal(sp.Rational(m, 2) - r)).is_zero()
            assert (bracket(LL(m), BB(r)) - BB(m + r).scal(sp.Rational(m, 2) - r)).is_zero()
            assert (bracket(JJ(m), GG(r)) - GG(m + r)).is_zero()
            assert (bracket(JJ(m), BB(r)) + BB(m + r)).is_zero()
            nchk += 4
    for r in HR:
        for s_ in HR:
            assert bracket(GG(r), GG(s_)).is_zero()
            assert bracket(BB(r), BB(s_)).is_zero()
            got = bracket(GG(r), BB(s_))
            tgt = LL(int(r + s_)).scal(2) + JJ(int(r + s_)).scal(r - s_)
            assert (got - tgt).is_zero(), (r, s_, got, tgt)
            nchk += 3
    print(f"      all of them hold identically             {nchk} checks   OK")

    # -- is the residue pairing the invariant one? -------------------------
    print()
    print("   the residue pairing <g,rho> = Res(g.rho) between the adjoint and")
    print("   the weight-1 densities is g-invariant:")
    els0 = [LL(m) for m in (-2, -1, 0, 1, 2)] + [JJ(m) for m in (-1, 0, 1)] \
        + [GG(sp.Rational(-1, 2)), GG(sp.Rational(1, 2)),
           BB(sp.Rational(-1, 2)), BB(sp.Rational(3, 2))]
    dens = [F.mono(n, s) for n in range(-3, 3) for s in ((), (1,), (2,), (1, 2))]
    ninv = 0
    for f in els0:
        for g in els0:
            for r in dens:
                lhs = res(bracket(f, g) * r)
                rhs = res(g * rho_action(f, r)).__mul__(1) \
                    if False else res(g * rho_action(f, r))
                tot = lhs + (-1) ** (f.parity() * g.parity()) * rhs
                assert sp.expand(tot) == 0, (f, g, r, lhs, rhs)
                ninv += 1
    print(f"      <[f,g],rho> + (-1)^(p(f)p(g)) <g, delta_f rho> = 0   "
          f"{ninv} checks   OK")

    # -- which sign convention makes c a 2-cocycle? ------------------------
    print()
    print("   the identification of densities with g^* is fixed by requiring")
    print("   that c be a 2-cocycle; the four candidate twists:")
    testels = [LL(m) for m in (-2, -1, 0, 1, 2)] + [JJ(m) for m in (-1, 0, 1)] \
        + [GG(sp.Rational(-1, 2)), GG(sp.Rational(1, 2)), GG(sp.Rational(3, 2)),
           BB(sp.Rational(-1, 2)), BB(sp.Rational(1, 2)), BB(sp.Rational(-3, 2))]
    good = []
    for tw in ("1", "p(f)", "p(g)", "p(f)p(g)"):
        globals()["SIGN"] = tw
        skew = all(sp.expand(coc(f, g) + (-1) ** (f.parity() * g.parity())
                               * coc(g, f)) == 0
                   for f in testels for g in testels)
        cyc = True
        for f, g, h in itertools.product(testels, repeat=3):
            pf, pg, ph = f.parity(), g.parity(), h.parity()
            tot = ((-1) ** (pf * ph) * coc(bracket(f, g), h)
                   + (-1) ** (pg * pf) * coc(bracket(g, h), f)
                   + (-1) ** (ph * pg) * coc(bracket(h, f), g))
            if sp.expand(tot) != 0:
                cyc = False
                break
        print(f"      twist (-1)^[{tw:<8}]:  super-skew {str(skew):<5} "
              f" cocycle identity {cyc}")
        if skew and cyc:
            good.append(tw)
    assert good, "no sign convention works"
    globals()["SIGN"] = good[0]
    print(f"   ==> taking the twist (-1)^[{good[0]}]")

    # -- the cocycle -------------------------------------------------------
    print()
    print("   the cocycle c(f,g) = Res( g . A(f) ),  A(f) = - d D1 D2 f:")
    print()
    vals = {}
    print("      Virasoro sector:")
    for m in range(-3, 4):
        v = coc(LL(m), LL(-m))
        pred = sp.Rational(m ** 3 - m, 12)
        cc = sp.simplify(v / pred) if pred != 0 else None
        print(f"         c(L_{m:>2}, L_{-m:>2}) = {v}"
              + (f"   =>  c = {cc}" if cc is not None else "   (m^3-m = 0)"))
        if cc is not None:
            vals.setdefault("Virasoro", set()).add(sp.simplify(cc))
    print("      u(1)-current sector:")
    for m in range(1, 4):
        v = coc(JJ(m), JJ(-m))
        cc = sp.simplify(v / sp.Rational(m, 3))
        print(f"         c(J_{m:>2}, J_{-m:>2}) = {v}   =>  c = {cc}")
        vals.setdefault("u(1)", set()).add(sp.simplify(cc))
    print("      supercharge sector:")
    for k in range(0, 3):
        r = sp.Rational(2 * k + 1, 2)
        v = coc(GG(r), BB(-r))
        pred = sp.Rational(1, 3) * (r ** 2 - sp.Rational(1, 4))
        cc = sp.simplify(v / pred) if pred != 0 else None
        print(f"         c(G_{r}, Gbar_{-r}) = {v}"
              + (f"   =>  c = {cc}" if cc is not None else "   (r^2-1/4 = 0)"))
        if cc is not None:
            vals.setdefault("supercharge", set()).add(sp.simplify(cc))

    print()
    for k, v in vals.items():
        print(f"      {k:<12} sector gives c in {v}")

    allv = set().union(*vals.values())
    print()
    print(f"   ==>  the three sectors agree:  c = {allv.pop() if len(allv)==1 else allv}")

    # -- the cocycle identity and skewness ---------------------------------
    print()
    els = [LL(m) for m in (-2, -1, 0, 1, 2)] + [JJ(m) for m in (-1, 0, 1)] \
        + [GG(sp.Rational(-1, 2)), GG(sp.Rational(1, 2)), GG(sp.Rational(3, 2)),
           BB(sp.Rational(-1, 2)), BB(sp.Rational(1, 2)), BB(sp.Rational(-3, 2))]
    nsk = 0
    for f in els:
        for g in els:
            assert sp.expand(coc(f, g) + (-1) ** (f.parity() * g.parity()) * coc(g, f)) == 0
            nsk += 1
    print(f"   super-skew-symmetry of c                      {nsk} checks   OK")
    ncy = 0
    for f, g, h in itertools.product(els, repeat=3):
        pf, pg, ph = f.parity(), g.parity(), h.parity()
        tot = ((-1) ** (pf * ph) * coc(bracket(f, g), h)
               + (-1) ** (pg * pf) * coc(bracket(g, h), f)
               + (-1) ** (ph * pg) * coc(bracket(h, f), g))
        assert sp.expand(tot) == 0, (f, g, h, tot)
        ncy += 1
    print(f"   the cocycle identity                          {ncy} checks   OK")
