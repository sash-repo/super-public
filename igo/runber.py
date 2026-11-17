"""Appendix A of the text: Berezin's obstructions (his "Introduction to
superanalysis", 2nd ed., Ch. 3, Secs. 4, 5, 7) against the Green--Manin class.

Part 1: exactness of the extension (22) through Euler characteristics over CP^1;
part 2: for odd dimension <= 2 the sub vanishes (Theorem A.5.1);
part 3: Lemma A.5.2 (Berezin's Example 1);
part 4: Proposition A.5.3 (a retractable non-split supermanifold, formula (24));
part 5: Statement A.5.4 (sdim of the vector fields, (12|9) against (8|6));
part 6: Lemma A.4.1 (the product (23) is minus the bracket, on 45 pairs);
part 7: Statement A.5.5 (the five classes, one representative of each);
part 8: Statement A.5.6 (none of the five is homogeneous).
"""

from fractions import Fraction
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import berobs as R                                                 # noqa: E402
import bashkin as B                                                # noqa: E402
import ns                                                          # noqa: E402
import sl2inv as S_                                                # noqa: E402
import orbit3 as O                                                 # noqa: E402

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "lib"))
from exactlin import rank as _rank                                 # noqa: E402


def main():
    print("=" * 78)
    print("1.  The extension  0 -> N (x) L^{q+1}N* -> T_q -> T (x) L^q N* -> 0")
    print("=" * 78)
    for ks in [(2, 2), (1, 2), (2, 2, 2), (1, 2, 3), (2, 2, 2, 2)]:
        for q in range(1, 2 * len(ks)):
            a = R.h0_h1(list(ks), q, "sub")
            b = R.h0_h1(list(ks), q, "quot")
            t = R.h0_h1(list(ks), q, "all")
            # exactness of the long exact sequence of a curve (H^2 = 0):
            # chi(T_q) = chi(sub) + chi(quot)
            R.check(f"ks={ks}, q={q}: chi(T_q) = chi(sub)+chi(quot)"
                    f"   [sub {a}, quot {b}, T_q {t}]",
                    t[0] - t[1], (a[0] - a[1]) + (b[0] - b[1]))

    print()
    print("=" * 78)
    print("2.  Odd dimension <= 2: the sub vanishes, so retraction == splitness")
    print("=" * 78)
    for ks in [(1,), (2,), (1, 1), (2, 2), (3, 5)]:
        m = len(ks)
        for q in range(2, 2 * m + 2):
            a = R.h0_h1(list(ks), q, "sub")
            R.check(f"ks={ks}, q={q}: sub = 0", a, (0, 0))
    print("        (Lambda^{q+1}N* = 0 for q >= 2 as soon as m <= 2, so every")
    print("         obstruction m_k of Sec. 7 vanishes identically there)")

    print()
    print("=" * 78)
    print("3.  Berezin's Example 1 of Sec. 6 recomputed (m = 2, N* = K + K)")
    print("=" * 78)
    ks = (2, 2)
    t = R.h0_h1(list(ks), 2, "all")
    R.check("ks=(2,2): (h^0, h^1)(T_2)", t, (0, 1))
    # his cocycle is  t_{U,V} = z^{-3} xi_1 ^ xi_2 d/dw; in the chart U, with
    # w = 1/x, d/dw = -x^2 d/dx, so it is  -x^{-1} xi_1 xi_2 d/dx = -u_{12} at
    # the exponent -1.
    coc = B.u((1, 2), -1, -1)
    R.check("his first obstruction n_1 is not a coboundary",
            R.is_coboundary(list(ks), 2, coc), False)
    R.check("  ... and it spans H^1(T_2)",
            B.in_H1_span(list(ks), 2, [coc])[1], True)

    print()
    print("=" * 78)
    print("4.  A RETRACTABLE NON-SPLIT supermanifold: the same, with a third")
    print("    odd coordinate  (X_0 = CP^1, N* = K + K + K, sdim = 1|3)")
    print("=" * 78)
    ks = (2, 2, 2)
    sub = R.h0_h1(list(ks), 2, "sub")
    quot = R.h0_h1(list(ks), 2, "quot")
    full = R.h0_h1(list(ks), 2, "all")
    R.check("h^1(N (x) L^3 N*) = 9 (Berezin's m_1 lives here)", sub[1], 9)
    R.check("h^1(T (x) L^2 N*) = 3 (his n_1 lives here)", quot[1], 3)
    R.check("h^0(T (x) L^2 N*) = 0, so H^1(sub) -> H^1(T_2) is injective",
            quot[0], 0)
    R.check("h^1(T_2) = 12 = 9 + 3 (the Green--Manin class)", full[1], 12)
    R.check("T_4 = 0, so Aut_(2) is abelian and [class] decides splitness",
            R.h0_h1(list(ks), 4, "all"), (0, 0))

    X = R.retractable_nonsplit(ks)
    print()
    print("    the supermanifold  " + X.name)
    print("      y      = " + _show(X.Y))
    for j, h in enumerate(X.H, start=1):
        print(f"      eta_{j}  = " + _show(h))
    # (a) it is retractable: the even coordinate is glued with NO odd term
    R.check("y is glued by the classical y = 1/x (a retraction exists)",
            X.Y, ns.g_from((), ns.lp((-1, 1))))
    R.check("  so every obstruction n_k of Secs. 4--5 vanishes",
            [k for k in (1, 2) if _n_k(X, k)], [])
    # (b) it is not split: the class is a non-zero element of H^1(T_2)
    coc = R.cocycle_of(ks)
    R.check("its class is not a coboundary in the sub",
            R.is_coboundary(list(ks), 2, coc, "sub"), False)
    R.check("its class is not a coboundary in T_2 either (NOT SPLIT)",
            R.is_coboundary(list(ks), 2, coc, "all"), False)
    R.check("its image in the quotient is zero (n_1 = 0)",
            R._keep(coc, "quot"), {})

    print()
    print("=" * 78)
    print("5.  Non-splitness once more, without any cohomology: the vector")
    print("    fields.  A split (CP^1, Lambda(K+K+K)) and X are compared.")
    print("=" * 78)
    S = ns.SM.split(list(ks), name="split(2,2,2)")
    ds = S.sdim_vect(degU=8)
    dx = X.sdim_vect(degU=8)
    print(f"        sdim v(split) = {ds},   sdim v(X) = {dx}")
    R.check("the two superdimensions differ, so X is not the split one",
            ds != dx, True)

    print()
    print("=" * 78)
    print("6.  The product of the last subsection of Sec. 7 IS the bracket")
    print("    of the corresponding derivations, which is why the second")
    print("    obstruction of Secs. 4--5 is shifted by the first one of Sec. 7")
    print("=" * 78)
    m = 5
    ok = True
    for (i, j) in [(1, 2), (2, 3), (1, 5)]:
        for t in (1, 2, 3, 4, 5):
            for J in [(1, 3, 4), (2, 3, 5), (3, 4, 5)]:
                lhs = S_.bracket_even(m, _u(m, (i, j)), _w(m, J, t))
                rhs = _berezin_circ(m, (i, j), J, t)
                if lhs[0] != ns.g_scal(rhs, -1) or any(lhs[1]):
                    ok = False
                    print(f"        mismatch at {(i, j)}, {J}, {t}")
    R.check("[u_{ij}, w_{J,t}] = - (u_{ij} o w_{J,t}) on 45 pairs, and the "
            "bracket has no d/dxi part", ok, True)

    print()
    print("=" * 78)
    print("7.  How many retractable non-split ones there are over K + K + K:")
    print("    H^1(sub) = V (x) W with dim V = dim W = 3, V the 3-dimensional")
    print("    sl(2)-module, so an isomorphism class is the rank together with")
    print("    the type of the invariant quadratic form on the image in V")
    print("=" * 78)
    G, _ = O.invariant_form()
    mats = {"(1, 'deg')": [[1, 2, -1], [0, 0, 0], [0, 0, 0]],
            "(1, 'nondeg')": [[1, 0, 3], [2, 0, 6], [-1, 0, -3]],
            "(2, 'deg')": [[1, 0, 0], [0, 2, 5], [0, 0, 0]],
            "(2, 'nondeg')": [[1, 1, 0], [0, 1, 1], [1, 2, 1]],
            "(3, '-')": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    sdims = {}
    for want, M in mats.items():
        inv = O.invariant(M, G)
        R.check(f"the class {M} is of type {want}", str(inv), want)
        sdims[want] = O.sm_of(M).sdim_vect(degU=8)
        print(f"        {want:14s} sdim v = {sdims[want]}")
    R.check("the five types are told apart by sdim v", len(set(sdims.values())),
            5)
    R.check("all five are smaller than the split (12, 9)",
            all(s < (12, 9) for s in sdims.values()), True)
    print("        (the census over a sample of 60 is in orbit3.py)")

    print()
    print("=" * 78)
    print("8.  None of the five is homogeneous (the split model is)")
    print("=" * 78)
    R.check("the split (CP^1, Lambda(K+K+K)) is homogeneous",
            ns.SM.split([2, 2, 2]).transitive(degU=8), True)
    for want, M in mats.items():
        R.check(f"the class of type {want} is not homogeneous",
                O.sm_of(M).transitive(degU=8), False)

    print()
    print("=" * 78)
    print(f"{R.PASS['n'] - R.PASS['bad']}/{R.PASS['n']} checks pass")
    print("=" * 78)
    return R.PASS["bad"]


def _u(m, I):
    """The derivation xi_I d/dx, a local section of T (x) Lambda^{|I|}N*."""
    return (ns.g_from(I, ns.lp((0, 1))), [ns.g_zero() for _ in range(m)])


def _w(m, J, t):
    """The derivation xi_J d/dxi_t, a local section of N (x) Lambda^{|J|}N*."""
    Bs = [ns.g_zero() for _ in range(m)]
    Bs[t - 1] = ns.g_from(J, ns.lp((0, 1)))
    return (ns.g_zero(), Bs)


def _berezin_circ(m, I, J, t):
    """(d/dx (x) xi_i ^ xi_j) o (xi'_t (x) eta), his formula
       (t (x) xi_1 ^ xi_2) o (xi' (x) eta)
              = t (x) ( xi'(xi_1) eta ^ xi_2 + xi'(xi_2) xi_1 ^ eta ),
    with eta = xi_J and xi' the basis vector of N dual to xi_t."""
    i, j = I
    out = ns.g_zero()
    if t == i:
        out = ns.g_add(out, ns.g_mul(ns.g_from(J, ns.lp((0, 1))),
                                     ns.g_from((j,), ns.lp((0, 1)))))
    if t == j:
        out = ns.g_add(out, ns.g_mul(ns.g_from((i,), ns.lp((0, 1))),
                                     ns.g_from(J, ns.lp((0, 1)))))
    return out


def _n_k(X, k):
    """The u-part of the glueing in degree 2k: Berezin's n_k cocycle."""
    out = {}
    for S, f in X.Y.items():
        if len(S) == 2 * k:
            out[S] = f
    return out


def _show(el):
    out = []
    for S, f in sorted(el.items()):
        terms = " + ".join(
            (f"{c}" if not e else (f"{c}x^{{{e}}}" if c != 1 else f"x^{{{e}}}"))
            for e, c in sorted(f.items()))
        xi = "".join(f"xi_{i}" for i in S)
        out.append(f"({terms}){xi}" if xi else f"{terms}")
    return " + ".join(out) or "0"


if __name__ == "__main__":
    sys.exit(main())
