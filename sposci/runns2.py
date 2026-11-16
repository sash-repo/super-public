#!/usr/bin/env python3
r"""
runns2.py -- driver for ns2.py.  Answers the questions:

STAGE 0  the two algebras, the 1-form conventions and the modules, all
         verified from scratch;
STAGE 1  the degree-0 part g_0 of each algebra, whence the NUMBER of
         parameters of the weighted densities: two over k^L(1|2), one over
         k^M(1|2);
STAGE 2  the analog of Proposition 7.3: which cF_{lambda, nu; mu},
         resp. cF^M_{lambda; mu}, carries an invariant bilinear form;
STAGE 3  the duality (cF_{lambda, nu; mu})^# = ?;
STAGE 4  Spin(W) for W = V (+) V^#: the central charge c, the weight h of the
         vacuum and the label H;
STAGE 5  c, h, H as exact functions of the parameters.
"""

from fractions import Fraction as Q
from itertools import product
import sys

from ns2 import (XI, ETA, THETA, ZETA, SUBSETS, mono, keys, add, smul, mul,
                 parity, clean, d_t, d_odd, cE, nabla, D_M,
                 br_L, br_M, K_L, K_M, act_L, act_M,
                 ALPHA1, FVARS, Lie1, dxi_multiplier,
                 jacobi_bad, module_bad, invariant_forms, form_report,
                 multiplier_space, degL, degM,
                 Wspace, hatX, cocycle, test_hat, VAC)

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))
from exactlin import rank as exrank, solve as exsolve      # noqa: E402

CZ, CT = Q(1), Q(1)                       # the constants of eq. (12a)


def actM(par, f, u):
    return act_M(par, f, u, CZ, CT)


def brM(f, g):
    return br_M(f, g, CZ, CT)


def nm(k):
    s = "".join({1: "a", 2: "b"}[i] for i in k[1])
    return "t^%s%s" % (k[0], s)


# ============================================================== STAGE 0

def stage0():
    print("=" * 74)
    print("STAGE 0.  The algebras, the 1-form conventions, and the modules")
    print("=" * 74)
    ks = keys(range(-2, 3))
    print("  k^L(1|2): Jacobi failures on %d^3 monomials: %d"
          % (len(ks), jacobi_bad(br_L, ks)))
    print("  k^M(1|2): Jacobi failures on %d^3 monomials: %d"
          % (len(ks), jacobi_bad(brM, ks)))

    # the 1-form conventions
    tests = [(a, S) for a in range(-2, 4) for S in SUBSETS]
    bad = 0
    for k in tests:
        f = mono(*k)
        L = Lie1(f, ALPHA1)
        want = {v: mul(smul(2, d_t(f)), ALPHA1[v]) for v in FVARS}
        for v in FVARS:
            if add(L.get(v, {}), smul(-1, want[v])):
                bad += 1
    print("  L_{K_f}(alpha_1) = 2 d_t(f) alpha_1 : %d defects over %d f"
          % (bad, len(tests)))
    OMS = [{"t": mono(1), XI: mono(0, (XI,)), ETA: mono(-1)},
           {"t": mono(0, (1, 2)), XI: mono(2, (ETA,)), ETA: mono(0)}]
    bad = 0
    gg = [(a, S) for a in range(-1, 3) for S in SUBSETS]
    for OM in OMS:
        for kf, kg in product(gg, repeat=2):
            f, g = mono(*kf), mono(*kg)
            lhs, rhs = Lie1(f, Lie1(g, OM)), Lie1(g, Lie1(f, OM))
            com = {v: add(lhs.get(v, {}),
                          smul(-((-1) ** (parity(f) * parity(g))), rhs.get(v, {})))
                   for v in FVARS}
            br = br_L(f, g)
            want = Lie1(br, OM) if br else {v: {} for v in FVARS}
            for v in FVARS:
                if add(com.get(v, {}), smul(-1, want.get(v, {}))):
                    bad += 1
    print("  L_{[K_f, K_g]} = [L_{K_f}, L_{K_g}] on 1-forms: %d defects" % bad)

    print("\n  The multiplier m(f) of [d xi], L_{K_f}[d xi] = m(f)[d xi]:")
    for k in [(1, ()), (0, (1, 2)), (0, ()), (2, ()), (1, (1,)), (1, (2,)),
              (1, (1, 2)), (-1, ())]:
        m = dxi_multiplier(mono(*k))
        print("      m(%-8s) = %s" % (nm(k), fmt(m)))

    print("\n  Module axiom [K_f, K_g].u = K_{f,g}.u:")
    ks, us = keys(range(-1, 2)), keys(range(-1, 2))
    tot = 0
    grid = [(Q(l, 3), Q(n, 2), Q(m, 4))
            for l in (-1, 0, 2) for n in (-1, 0, 3) for m in (0, 1, 3)]
    for par in grid:
        tot += module_bad(act_L, br_L, par, ks, us)
    print("      k^L(1|2): %d (lambda, nu, mu) triples, %d failures"
          % (len(grid), tot))
    gridM = [(Q(l, 3), Q(m, 4)) for l in (-1, 0, 2, 5) for m in (0, 1, 2, 3)]
    tot = 0
    for par in gridM:
        tot += module_bad(actM, brM, par, ks, us)
    print("      k^M(1|2): %d (lambda, mu) pairs, %d failures"
          % (len(gridM), tot))


def fmt(f):
    if not f:
        return "0"
    return " + ".join("%s %s" % (c, nm(k)) for k, c in sorted(f.items()))


# ============================================================== STAGE 1

def stage1():
    print("\n" + "=" * 74)
    print("STAGE 1.  g_0, and the number of parameters of the densities")
    print("=" * 74)
    print("  k^L(1|2)_0 is spanned by K_t and K_{xi eta}:")
    print("      {t, xi eta}      = %s" % fmt(br_L(mono(1), mono(0, (XI, ETA)))))
    print("      {xi eta, xi eta} = %s"
          % fmt(br_L(mono(0, (XI, ETA)), mono(0, (XI, ETA)))))
    print("      both generators are EVEN, so g_0 is a commutative Lie algebra")
    print("      of dim 2 -- two characters, whence cF_{lambda, nu; mu}.")
    print("  k^M(1|2)_0 is spanned by K^M_t (even) and K^M_{t zeta} (odd):")
    print("      {t, t zeta}_R      = %s" % fmt(brM(mono(1), mono(1, (ZETA,)))))
    print("      {t zeta, t zeta}_R = %s"
          % fmt(brM(mono(1, (ZETA,)), mono(1, (ZETA,)))))
    print("      so g_0 has sdim (1|1) and is the odd Heisenberg superalgebra;")
    print("      its 1-dimensional modules are trivial and its irreducible")
    print("      (1|1)-dimensional ones form ONE 1-parameter family, whence")
    print("      cF^M_{lambda; mu} and no second parameter.")

    for name, Kfun, br, dg in (
            ("k^L(1|2)", lambda f, u: K_L(f, u), br_L, degL),
            ("k^M(1|2)", lambda f, u: K_M(f, u, Q(0), CZ, CT), brM, degM)):
        for N, D in ((4, 3), (5, 4)):
            dc, dcob, _, _ = multiplier_space(Kfun, br, N=N, D=D, deg=dg)
            print("  %s  N=%d D=%d: dim{1-cocycles cF-valued, degree 0} = %d, "
                  "dim{coboundaries} = %d  =>  %d parameters (mu included)"
                  % (name, N, D, dc, dcob, dc - dcob))


# ============================================================== STAGE 2

def stage2(N=5):
    print("\n" + "=" * 74)
    print("STAGE 2.  Invariant forms on the module itself (analog of Proposition 7.3)")
    print("=" * 74)
    print("\n  --- k^L(1|2), cF_{lambda, nu; mu} ---")
    hitsL = []
    for lam in [Q(l, 2) for l in range(-2, 5)]:
        for nu in [Q(n, 2) for n in range(-2, 5)]:
            for mu in (Q(0), Q(1, 2), Q(1, 3)):
                forms, _ = invariant_forms(act_L, (lam, nu, mu), (lam, nu, mu), N=N)
                for form in forms:
                    par, eps, nd, _ = form_report(form, N)
                    hitsL.append((lam, nu, mu, par, nd))
                    print("      lambda=%-5s nu=%-5s mu=%-5s dim=%d parity=%s "
                          "eps=%s nondeg=%s"
                          % (lam, nu, mu, len(forms), par,
                             sorted(map(str, eps)), nd))
    if not hitsL:
        print("      no invariant form for any (lambda, nu, mu) on the grid")
    print("\n  --- k^M(1|2), cF^M_{lambda; mu} ---")
    hitsM = []
    for lam in [Q(l, 2) for l in range(-2, 5)]:
        for mu in (Q(0), Q(1, 4), Q(1, 2), Q(3, 4), Q(1, 3)):
            forms, _ = invariant_forms(actM, (lam, mu), (lam, mu), N=N)
            for form in forms:
                par, eps, nd, _ = form_report(form, N)
                hitsM.append((lam, mu, par, nd))
                print("      lambda=%-5s mu=%-5s dim=%d parity=%s eps=%s nondeg=%s"
                      % (lam, mu, len(forms), par, sorted(map(str, eps)), nd))
    if not hitsM:
        print("      no invariant form for any (lambda, mu) on the grid")
    return hitsL, hitsM


# ============================================================== STAGE 3

def stage3(N=5):
    print("\n" + "=" * 74)
    print("STAGE 3.  The dual module")
    print("=" * 74)
    print("\n  --- k^L(1|2) ---")
    for lam, nu, mu in [(Q(1, 2), Q(1, 3), Q(1, 5)), (Q(0), Q(0), Q(0)),
                        (Q(2), Q(-1), Q(1, 2)), (Q(1), Q(1), Q(0))]:
        found = []
        for l2 in [Q(k, 2) for k in range(-6, 7)]:
            for n2 in [Q(k, 3) for k in range(-6, 7)]:
                for m2 in (-mu, Q(1, 2) - mu):
                    forms, _ = invariant_forms(act_L, (lam, nu, mu), (l2, n2, m2),
                                               N=N)
                    for form in forms:
                        par, eps, nd, _ = form_report(form, N)
                        found.append((str(l2), str(n2), str(m2), "par=%s" % par,
                                      "nondeg" if nd else "DEG"))
        print("      (%s, %s; %s)^# pairs with %s" % (lam, nu, mu, found))
    print("\n  --- k^M(1|2) ---")
    for lam, mu in [(Q(1, 2), Q(1, 5)), (Q(0), Q(0)), (Q(2), Q(1, 2)),
                    (Q(1), Q(1, 4))]:
        found = []
        for l2 in [Q(k, 2) for k in range(-6, 7)]:
            for m2 in (-mu, Q(1, 2) - mu, -Q(1, 2) - mu, Q(1, 4) - mu,
                       Q(3, 4) - mu):
                forms, _ = invariant_forms(actM, (lam, mu), (l2, m2), N=N)
                for form in forms:
                    par, eps, nd, _ = form_report(form, N)
                    found.append((str(l2), str(m2), "par=%s" % par,
                                  "nondeg" if nd else "DEG"))
        print("      (%s; %s)^# pairs with %s" % (lam, mu, found))


# ============================================================== STAGE 4

def build_W(sector, par, N, dualrule):
    act = act_L if sector == "L" else actM
    par_d = dualrule(par)
    forms, _ = invariant_forms(act, par, par_d, N=N)
    if not forms:
        return None
    ev = [f for f in forms
          if all((len(ku[1]) + len(kv[1])) % 2 == 0 for (ku, kv) in f)
          and form_report(f, N)[2]]
    od = [f for f in forms
          if all((len(ku[1]) + len(kv[1])) % 2 == 1 for (ku, kv) in f)
          and form_report(f, N)[2]]
    if ev:
        return Wspace(act, par, par_d, ev[0], par_shift=0)
    if od:
        return Wspace(act, par, par_d, od[0], par_shift=1)
    return None


def LL(m):
    return mono(m + 1)


def alpha_from_LL(W, bracket, win, ms=(2, 3, 4)):
    """c(K_{t^{m+1}}, K_{t^{1-m}}) = alpha m^3 + kappa m;  c_central = 3 alpha."""
    vals = []
    for m in ms:
        v, rest = cocycle(W, bracket, LL(m), LL(-m), win)
        assert not rest, "not central: %s" % rest
        vals.append((m, v))
    (m1, v1), (m2, v2) = vals[0], vals[1]
    det = m1 ** 3 * m2 - m1 * m2 ** 3
    al = (v1 * m2 - m1 * v2) / det
    ka = (m1 ** 3 * v2 - v1 * m2 ** 3) / det
    for m, v in vals[2:]:
        assert al * m ** 3 + ka * m == v, ("c(L_m, L_-m) is not a cubic", vals)
    return al


def h_of(W, win):
    """h = -(1/8) <hat K_{t^2} hat K_1>, since [ell_1, ell_{-1}] = -2 ell_0
    carries no central term."""
    st = hatX(W, LL(1), hatX(W, LL(-1), VAC, win), win)
    return -Q(1, 8) * st.get((), Q(0))


def H_of(W, win):
    """the eigenvalue of hat K_{xi eta} on the vacuum; K_{xi eta} is even, and
    the operator is the quadratic element of eq. (150) taken as it stands."""
    st = hatX(W, mono(0, (XI, ETA)), VAC, win)
    rest = {k: v for k, v in st.items() if k != ()}
    assert not rest, "hat K_{xi eta} does not fix the vacuum: %s" % rest
    return st.get((), Q(0))


def stage4(N=6, wins=(4, 5)):
    print("\n" + "=" * 74)
    print("STAGE 4.  Spin(W) for W = V (+) V^#")
    print("=" * 74)
    print("  the defining property [hat X, psi_v] = psi_{Xv}, failures:")
    for sector, par, rule in (("L", (Q(1, 3), Q(1, 5), Q(1, 7)), DUAL_L),
                              ("M", (Q(1, 3), Q(1, 5)), DUAL_M)):
        W = build_W(sector, par, N, rule)
        fs = [mono(a, S) for a in (-1, 0, 1, 2) for S in SUBSETS
              if len(S) % 2 == 0]
        print("      %s (even elements): %d" % (sector, test_hat(W, wins[-1], fs)))

    print("\n%-3s %-8s %-8s %-8s %-10s %-10s %-10s %s"
          % ("sec", "lambda", "nu", "mu", "c", "h", "H", "windows"))
    print("-" * 74)
    out = {}
    for par in ((Q(1, 3), Q(1, 5), Q(1, 7)), (Q(0), Q(0), Q(0)),
                (Q(1), Q(0), Q(0)), (Q(2), Q(-1), Q(1, 2)),
                (Q(-3, 7), Q(5, 2), Q(2, 9)), (Q(1, 2), Q(1, 2), Q(1, 3))):
        W = build_W("L", par, N, DUAL_L)
        if W is None:
            print("L   %-8s %-8s %-8s  no invariant pairing" % par)
            continue
        res = [(3 * alpha_from_LL(W, br_L, w), h_of(W, w), H_of(W, w))
               for w in wins]
        print("L   %-8s %-8s %-8s %-10s %-10s %-10s %s"
              % (par[0], par[1], par[2], res[0][0], res[0][1], res[0][2],
                 "stable" if res[0] == res[1] else "UNSTABLE %s" % (res,)))
        out[("L",) + par] = res[0]
    for par in ((Q(1, 3), Q(1, 7)), (Q(0), Q(0)), (Q(1), Q(0)),
                (Q(2), Q(1, 2)), (Q(-3, 7), Q(2, 9)), (Q(1, 2), Q(1, 3))):
        W = build_W("M", par, N, DUAL_M)
        if W is None:
            print("M   %-8s %-8s %-8s  no invariant pairing" % (par[0], "-", par[1]))
            continue
        res = [(3 * alpha_from_LL(W, brM, w), h_of(W, w)) for w in wins]
        print("M   %-8s %-8s %-8s %-10s %-10s %-10s %s"
              % (par[0], "-", par[1], res[0][0], res[0][1], "n/a",
                 "stable" if res[0] == res[1] else "UNSTABLE %s" % (res,)))
        out[("M",) + par] = res[0]
    return out


def DUAL_L(par):
    lam, nu, mu = par
    return (-lam, -nu, -mu)


def DUAL_M(par):
    lam, mu = par
    return (-lam, Q(1, 2) - mu)


# ============================================================== STAGE 5

def fit(points, monoms, names):
    A = [[m(*p[0]) for m in monoms] for p in points]
    b = [p[1] for p in points]
    if exrank(A) < len(monoms):
        return None
    from itertools import combinations
    for rows in combinations(range(len(A)), len(monoms)):
        sub = [A[i] for i in rows]
        if exrank(sub) < len(monoms):
            continue
        x = exsolve(sub, [b[i] for i in rows])
        if x is None:
            continue
        if all(sum(c * m(*p[0]) for c, m in zip(x, monoms)) == p[1]
               for p in points):
            return x
        return None
    return None


def stage5(N=6, win=4):
    print("\n" + "=" * 74)
    print("STAGE 5.  c, h and H as exact functions of the parameters")
    print("=" * 74)
    gridL = [(Q(l, 3), Q(n, 2), Q(m, 4))
             for l in (-1, 0, 2, 4) for n in (-1, 0, 1, 3) for m in (0, 1, 3)]
    monL = [lambda l, n, u: Q(1), lambda l, n, u: l, lambda l, n, u: n,
            lambda l, n, u: u, lambda l, n, u: l * l, lambda l, n, u: n * n,
            lambda l, n, u: u * u, lambda l, n, u: l * n, lambda l, n, u: l * u,
            lambda l, n, u: n * u]
    namL = ["1", "lambda", "nu", "mu", "lambda^2", "nu^2", "mu^2",
            "lambda.nu", "lambda.mu", "nu.mu"]
    pts = {"c": [], "h": [], "H": []}
    for par in gridL:
        W = build_W("L", par, N, DUAL_L)
        if W is None:
            print("  L %s: no pairing" % (par,))
            continue
        pts["c"].append((par, 3 * alpha_from_LL(W, br_L, win)))
        pts["h"].append((par, h_of(W, win)))
        pts["H"].append((par, H_of(W, win)))
    for what in ("c", "h", "H"):
        x = fit(pts[what], monL, namL)
        if x is None:
            print("  L  %s: no exact quadratic fit on %d points" % (what, len(pts[what])))
            for p in pts[what][:12]:
                print("       %s -> %s" % (p[0], p[1]))
        else:
            terms = " + ".join("%s*%s" % (c, n) for c, n in zip(x, namL) if c)
            print("  L  %s = %s   (exact on all %d points)"
                  % (what, terms if terms else "0", len(pts[what])))
    gridM = [(Q(l, 3), Q(m, 4)) for l in (-1, 0, 2, 4, 5) for m in (0, 1, 2, 3)]
    monM = [lambda l, u: Q(1), lambda l, u: l, lambda l, u: u,
            lambda l, u: l * l, lambda l, u: l * u, lambda l, u: u * u]
    namM = ["1", "lambda", "mu", "lambda^2", "lambda.mu", "mu^2"]
    ptsM = {"c": [], "h": [], "H": []}
    for par in gridM:
        W = build_W("M", par, N, DUAL_M)
        if W is None:
            print("  M %s: no pairing" % (par,))
            continue
        ptsM["c"].append((par, 3 * alpha_from_LL(W, brM, win)))
        ptsM["h"].append((par, h_of(W, win)))
    for what in ("c", "h"):
        x = fit(ptsM[what], monM, namM)
        if x is None:
            print("  M  %s: no exact quadratic fit on %d points" % (what, len(ptsM[what])))
            for p in ptsM[what][:12]:
                print("       %s -> %s" % (p[0], p[1]))
        else:
            terms = " + ".join("%s*%s" % (c, n) for c, n in zip(x, namM) if c)
            print("  M  %s = %s   (exact on all %d points)"
                  % (what, terms if terms else "0", len(ptsM[what])))


if __name__ == "__main__":
    which = sys.argv[1:] or ["0", "1", "2", "3", "4", "5"]
    if "0" in which:
        stage0()
    if "1" in which:
        stage1()
    if "2" in which:
        stage2()
    if "3" in which:
        stage3()
    if "4" in which:
        stage4()
    if "5" in which:
        stage5()
