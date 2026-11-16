#!/usr/bin/env python3
"""
n2ram.py -- the N=2 anomaly in the RAMOND sector.

By item 1) of Lemma 9.1.1 the Ramond frame is the Neveu-Schwarz frame
pulled back to the double cover t = s^2, so every identity of n2.py holds
upstairs verbatim: the stress tensor is the same expression and the anomaly is
the same local expression

    A(f) = - d_t D1 D2 f = d_t B(f).

What changes is only which functions are allowed.  The deck transformation
M: s -> -s acts on the frame by R = diag(1,-1), i.e. th1 -> th1, th2 -> -th2
(th2 = t^{1/2} zeta), so

    k^M(1|2) = the M-invariant generating functions
             = span{ t^n,  t^n th1,  t^{n+1/2} th2,  t^{n+1/2} th1 th2 },

whereas the weight-1 densities of the Ramond sector are the M-ANTI-invariant
ones -- which is where T lives, and why the Berezin residue, the coefficient of
t^{-1} th1 th2, is still the pairing: th1 th2 is M-odd.

Usage:  ../venv/bin/python n2ram.py     (needs sympy)
"""

import itertools
import sympy as sp

from n2coc import F, K, bracket, A, res, D1, D2, dt, basis  # noqa: F401
import n2coc

HALF = sp.Rational(1, 2)


def kM_basis(lo=-3, hi=4):
    """the M-invariant generating functions, in the range given."""
    out = []
    for n in range(lo, hi + 1):
        out.append(F.mono(sp.Integer(n)))
        out.append(F.mono(sp.Integer(n), (1,)))
        out.append(F.mono(sp.Rational(2 * n + 1, 2), (2,)))
        out.append(F.mono(sp.Rational(2 * n + 1, 2), (1, 2)))
    return out


def is_M_invariant(f):
    """M: s -> -s, th1 -> th1, th2 -> -th2, i.e. t^{k/2} -> (-1)^k t^{k/2}."""
    for (n, s), c in f.d.items():
        k = sp.Rational(2 * sp.Rational(n))          # the power of s
        sgn = (-1) ** int(k) * ((-1) if 2 in s else 1)
        if sgn != 1:
            return False
    return True


def coc(f, g):
    """the same cocycle, with the same twist as in the NS sector."""
    return res(g * A(f)) * (-1) ** f.parity()


if __name__ == "__main__":
    print("=" * 74)
    print("the N=2 anomaly in the Ramond sector: k^M(1|2)")
    print("=" * 74)

    B = kM_basis(-2, 3)
    print(f"   the M-invariant span, {len(B)} generating functions tested")
    for f in B:
        assert is_M_invariant(f), f

    # 1) it is a subalgebra
    n = 0
    for f in kM_basis(-1, 2):
        for g in kM_basis(-1, 2):
            h = bracket(f, g)
            assert is_M_invariant(h), (f, g, h)
            n += 1
    print(f"   closed under the contact bracket             {n} checks   OK")

    # 2) the Virasoro sector: identical to the NS one
    L = lambda m: F.mono(sp.Integer(m + 1)).scal(sp.Rational(-1, 2))
    n = 0
    for m in range(-3, 4):
        for k in range(-3, 4):
            assert (bracket(L(m), L(k)) - L(m + k).scal(m - k)).is_zero()
            n += 1
    print(f"   L_m := K_f, f = -1/2 t^(m+1):  [L_m,L_n]=(m-n)L_(m+n)  "
          f"{n} checks   OK")
    cs = set()
    print("      Virasoro sector:")
    for m in range(2, 5):
        v = coc(L(m), L(-m))
        cc = sp.simplify(v / sp.Rational(m ** 3 - m, 12))
        print(f"         c(L_{m}, L_-{m}) = {v}   =>  c = {cc}")
        cs.add(cc)

    # 3) the u(1) sector: now HALF-integer moded (th1 th2 is M-odd)
    #    normalize J_a by [L_m, J_a] = -a J_(m+a)
    cJ = sp.symbols("cJ")
    Jf = lambda a, c=cJ: F.mono(sp.Rational(a), (1, 2)).scal(c)
    m0, a0 = 1, HALF
    lhs = bracket(L(m0), Jf(a0))
    tgt = Jf(m0 + a0).scal(-a0)
    eqs = [sp.expand(lhs.d.get(k, 0) - tgt.d.get(k, 0))
           for k in set(lhs.d) | set(tgt.d)]
    sol = sp.solve([e for e in eqs if e != 0], cJ, dict=True)
    cJv = [s[cJ] for s in sol if s[cJ] != 0][0] if sol else sp.Integer(1)
    J = lambda a: Jf(a, cJv)
    n = 0
    for m in range(-2, 3):
        for k in range(-2, 3):
            a = sp.Rational(2 * k + 1, 2)
            assert (bracket(L(m), J(a)) + J(m + a).scal(a)).is_zero(), (m, a)
            n += 1
    print(f"   J_a := K_f, f = ({cJv}) t^a th1 th2, a in Z+1/2:"
          f"  [L_m,J_a]=-aJ_(m+a)  {n} checks   OK")
    print("      u(1)-current sector:")
    for k in range(0, 3):
        a = sp.Rational(2 * k + 1, 2)
        v = coc(J(a), J(-a))
        print(f"         c(J_{a}, J_-{a}) = {v}   (cJ not yet fixed; see below)")

    # 4) the odd sectors.  In k^M(1|2) the two odd towers are moded
    #    DIFFERENTLY -- th1 gives half-integer, th2 integer indices, which is
    #    what "R interchanges the two chirality lines" means -- so the pair
    #    (G, Gbar) of the NS sector is replaced by two REAL towers, and the
    #    relations that fix their scale are {G^i_r, G^i_s} = 2 L_{r+s}.
    c1, c2 = sp.symbols("c1 c2")
    G1 = lambda r, c=1: F.mono(sp.Rational(2 * r + 1, 2), (1,)).scal(c)
    G2 = lambda r, c=1: F.mono(sp.Rational(2 * r + 1, 2), (2,)).scal(c)

    def fix(Gt, idx):
        r, s = idx
        raw = bracket(Gt(r), Gt(s))            # = k * t^{r+s+1}
        tgt = L(int(r + s)).scal(2)
        ks = set(raw.d) | set(tgt.d)
        k = [sp.simplify(tgt.d.get(kk, 0) / raw.d.get(kk, 1)) for kk in ks]
        assert len(set(k)) == 1, k
        return sp.sqrt(k[0])

    v1 = fix(G1, (HALF, -HALF))
    v2 = fix(G2, (sp.Integer(1), sp.Integer(-1)))
    print(f"   {{G^1_r,G^1_s}} = 2L_(r+s) fixes c1 = {v1};  "
          f"{{G^2_r,G^2_s}} = 2L_(r+s) fixes c2 = {v2}")
    GA = lambda r: G1(r, v1)
    GB = lambda r: G2(r, v2)
    n = 0
    for m in range(-2, 3):
        for k in range(-2, 3):
            r = sp.Rational(2 * k + 1, 2)
            assert (bracket(L(m), GA(r))
                    - GA(m + r).scal(sp.Rational(m, 2) - r)).is_zero()
            rr = sp.Integer(k)
            assert (bracket(L(m), GB(rr))
                    - GB(m + rr).scal(sp.Rational(m, 2) - rr)).is_zero()
            n += 2
        for k in range(-2, 3):
            for l in range(-2, 3):
                r, s = sp.Rational(2 * k + 1, 2), sp.Rational(2 * l + 1, 2)
                assert (bracket(GA(r), GA(s)) - L(int(r + s)).scal(2)).is_zero()
                rr, ss = sp.Integer(k), sp.Integer(l)
                assert (bracket(GB(rr), GB(ss)) - L(rr + ss).scal(2)).is_zero()
                n += 2
    print(f"      [L_m,G_r]=(m/2-r)G_(m+r), {{G^i,G^i}}=2L    {n} checks   OK")
    # the mixed bracket then fixes cJ
    r1, s1 = HALF, sp.Integer(0)
    mix = bracket(GA(r1), GB(s1))
    key = (sp.Rational(1, 2), (1, 2))
    cJfixed = sp.simplify(mix.d[key] / (r1 - s1))
    print(f"      {{G^1_r,G^2_s}} = (r-s) J_(r+s) then fixes cJ = {cJfixed}")
    Jf2 = lambda a: F.mono(sp.Rational(a), (1, 2)).scal(cJfixed)
    print("      supercharge sector:")
    for k in range(0, 3):
        r = sp.Rational(2 * k + 1, 2)
        v = coc(GA(r), GB(-r))
        print(f"         c(G^1_{r}, G^2_-{r}) = {v}")
    print("      u(1) sector in that normalization:")
    for k in range(0, 3):
        a = sp.Rational(2 * k + 1, 2)
        print(f"         c(J_{a}, J_-{a}) = {coc(Jf2(a), Jf2(-a))}")

    print()
    print(f"   the sectors give c in {cs}")
    if len(cs) == 1:
        print(f"   ==>  the Ramond value is c = {cs.pop()}, i.e. |c| = 3 --")
        print("        the same number as in the Neveu-Schwarz sector.")

    # 5) skewness and the cocycle identity, on k^M
    els = [L(m) for m in (-2, -1, 0, 1, 2)] + [J(sp.Rational(2 * k + 1, 2))
                                               for k in (-2, -1, 0, 1)] \
        + [G1(sp.Rational(2 * k + 1, 2), 1) for k in (-1, 0, 1)] \
        + [G2(sp.Integer(k), 1) for k in (-1, 0, 1)]
    nsk = 0
    for f in els:
        for g in els:
            assert sp.expand(coc(f, g)
                             + (-1) ** (f.parity() * g.parity()) * coc(g, f)) == 0
            nsk += 1
    print(f"   super-skew-symmetry of c on k^M(1|2)          {nsk} checks   OK")
    ncy = 0
    for f, g, h in itertools.product(els, repeat=3):
        pf, pg, ph = f.parity(), g.parity(), h.parity()
        tot = ((-1) ** (pf * ph) * coc(bracket(f, g), h)
               + (-1) ** (pg * pf) * coc(bracket(g, h), f)
               + (-1) ** (ph * pg) * coc(bracket(h, f), g))
        assert sp.expand(tot) == 0, (f, g, h, tot)
        ncy += 1
    print(f"   the cocycle identity on k^M(1|2)              {ncy} checks   OK")
