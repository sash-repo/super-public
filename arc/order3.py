#!/usr/bin/env python3
"""Two checks about the factors of a Schwarzian.

 1) Is the order of the factors important in the row
    s = D_{theta_1} D_{theta_2} D_1 of eq. (205) on k^L(1|2).  We run all
    six orderings of the three factors and compare the resulting cocycles
    c(f,g) = (-1)^{p(f)} Res(f s(g)).

 2) The operator D_xi D_eta d_t of item b) of Subsec. 4.5.2 of the text, in the xi-eta coordinates of eq. (28).  We check, on
    k^L(1|2) realized in xi, eta:
       (a)  D_xi D_eta d_t taken literally is NOT a cocycle;
       (b)  its antisymmetrization (D_xi D_eta - d_t) d_t IS one, with the
            factor (-1)^{p(f)};
       (c)  the same without the sign factor is not.

    python3 order3.py > order3.log
"""
from itertools import permutations, combinations
from fractions import Fraction as F

from stringy import mono, add, mul, smul, d_t, d_odd, parity, eq, show
from check_cocycles import L_bracket, L_res, L_ops, check_cocycle_identity


def compose(*ops):
    def h(f):
        for op in reversed(ops):
            f = op(f)
        return f
    return h


def keys(n, arange=range(-2, 3)):
    out = []
    labels = tuple(range(1, n + 1))
    for a in arange:
        for r in range(n + 1):
            for S in combinations(labels, r):
                out.append((a, S))
    return out


def coc(sn, sign=True):
    def c(f, g):
        v = L_res(2, mul(f, sn(g)))
        return (-1) ** parity(f) * v if sign else v
    return c


# ------------------------------------------------------------------ theta basis
def q1_orderings():
    n = 2
    D1, Dth, Kth = L_ops(n)
    names = {"D_th1": Dth(1), "D_th2": Dth(2), "D_1": D1}
    ks = keys(n)
    small = keys(n, range(-2, 3))
    print("=== 1) the six orderings of D_th1, D_th2, D_1 on k^L(1|2)")
    ref = None
    for perm in permutations(["D_th1", "D_th2", "D_1"]):
        sn = compose(*[names[p] for p in perm])
        c = coc(sn)
        bad = check_cocycle_identity(n, lambda f, g: L_bracket(n, f, g), c, small)
        vals = {(kf, kg): c(mono(*kf), mono(*kg)) for kf in ks for kg in ks}
        if ref is None:
            ref, refname = vals, perm
            rel = "reference"
        else:
            ratios = {v / ref[k] for k, v in vals.items() if v or ref[k]
                      if ref[k]}
            zeros_agree = all((v == 0) == (ref[k] == 0) for k, v in vals.items())
            rel = (f"= ({list(ratios)[0]}) x reference" if len(ratios) == 1
                   and zeros_agree else f"ratios {sorted(ratios)[:4]}")
        print(f"  {' '.join(perm):22s} cocycle: "
              f"{'OK' if bad == 0 else str(bad) + ' failures':12s}  {rel}")
    print("  (the reference is", " ".join(refname), ")")


# ------------------------------------------------------------------ xi-eta basis
def q2_xieta():
    """k^L(1|2) in the isotropic coordinates (xi, eta) of eq. (28), i.e. for
    alpha_1 = dt + xi d eta + eta d xi:  D_xi, D_eta and the bracket are the
    ones of check_bases.py (the "hyperbolic realisation"), where
    {D_xi, D_eta} = 2 d_t."""
    from check_bases import L_bracket_h, D_xi, D_eta
    n, N = 1, 2                      # k^L(1|2): one (xi, eta) pair
    Dx, De = D_xi(1), D_eta(1)
    small = keys(N, range(-2, 3))
    br = lambda f, g: L_bracket_h(n, f, g)
    res = lambda f: L_res(N, f)

    print("\n=== 2) the operator D_xi D_eta d_t of item b), in (xi, eta)")
    ok = all(eq(add(Dx(De(mono(*k))), De(Dx(mono(*k)))), smul(2, d_t(mono(*k))))
             for k in small)
    print(f"   {{D_xi, D_eta}} = 2 d_t: {'OK' if ok else 'FAILS'}")

    def cocf(sn, sign):
        def c(f, g):
            v = res(mul(f, sn(g)))
            return (-1) ** parity(f) * v if sign else v
        return c

    lit = compose(Dx, De, d_t)
    anti = compose(lambda f: add(Dx(De(f)), smul(-1, d_t(f))), d_t)
    anti2 = compose(lambda f: smul(F(1, 2), add(Dx(De(f)), smul(-1, De(Dx(f))))),
                    d_t)
    for name, sn, sign in (
            ("D_xi D_eta d_t, literal, with (-1)^p(f)", lit, True),
            ("D_xi D_eta d_t, literal, no sign factor", lit, False),
            ("(D_xi D_eta - d_t) d_t, with (-1)^p(f)", anti, True),
            ("(D_xi D_eta - d_t) d_t, no sign factor", anti, False),
            ("1/2 (D_xi D_eta - D_eta D_xi) d_t, with (-1)^p(f)", anti2, True)):
        bad = check_cocycle_identity(N, br, cocf(sn, sign), small)
        print(f"   {name:52s} "
              f"{'COCYCLE' if bad == 0 else str(bad) + ' failures'}")

    ok = all(eq(anti(mono(*k)), anti2(mono(*k))) for k in small)
    print("   (D_xi D_eta - d_t) d_t = 1/2 (D_xi D_eta - D_eta D_xi) d_t: "
          f"{'OK' if ok else 'FAILS'}")


if __name__ == "__main__":
    q1_orderings()
    q2_xieta()
