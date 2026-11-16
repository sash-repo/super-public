#!/usr/bin/env python3
r"""
runns1.py -- driver for ns1.py.  Answers the questions:

STAGE 0  the two algebras and their modules, verified from scratch;
STAGE 1  the analog of Proposition 7.3: which cF_{lambda;mu} over
         ns = k^L(1|1) and over r = k^M(1|1) carry an invariant nondegenerate
         bilinear form, of which parity and which symmetry;
STAGE 2  the duality cF_{lambda;mu}^# = cF_{lambda^*;-mu};
STAGE 3  Spin(W) for W = cF_{lambda;mu} (+) cF_{lambda;mu}^#: the central
         charge c and the weight h of the vacuum.
"""

from fractions import Fraction as Q
from itertools import product
import sys

from ns1 import (mono, keys, add, smul, parity, jacobi_bad, module_bad,
                 br_NS, br_R, act_NS, act_R, invariant_forms, form_report,
                 Wspace, hatX, cocycle, test_hat, VAC)

GRID = [(Q(l, 2), Q(m, 2)) for l in range(-2, 5) for m in range(0, 2)]


def stage0():
    print("=" * 72)
    print("STAGE 0.  The algebras ns = k^L(1|1), r = k^M(1|1) and their modules")
    print("=" * 72)
    ks = keys(range(-2, 3))
    print("  NS: Jacobi failures on %d^3 monomials: %d" % (len(ks), jacobi_bad(br_NS, ks)))

    # the Ramond convention constants (s, kz), fixed by Jacobi
    good = []
    for s in (1, -1):
        for kz in (Q(1), Q(-1), Q(1, 2), Q(2)):
            bad = jacobi_bad(lambda f, g, s=s, kz=kz: br_R(f, g, s, kz), ks)
            if bad == 0:
                good.append((s, kz))
    print("  R : (s, kz) with 0 Jacobi failures: %s" % (good,))
    return good


def stage0b(s, kz):
    print("\n  Module axiom [K_f, K_g].u = K_{f,g}.u on cF_{lambda;mu}:")
    ks = keys(range(-1, 2))
    us = keys(range(-1, 2))
    for name, act, br in (("NS", act_NS, br_NS),
                          ("R", lambda l, m, f, u: act_R(l, m, f, u, s, kz),
                           lambda f, g: br_R(f, g, s, kz))):
        tot = 0
        for lam, mu in GRID:
            tot += module_bad(act, br, lam, mu, ks, us)
        print("    %-3s %d (lambda, mu) pairs, %d failures" % (name, len(GRID), tot))


def stage1(s, kz, N=5):
    print("\n" + "=" * 72)
    print("STAGE 1.  Invariant forms on cF_{lambda;mu} itself  (analog of Proposition 7.3)")
    print("=" * 72)
    out = {}
    for name, act in (("NS", act_NS),
                      ("R", lambda l, m, f, u: act_R(l, m, f, u, s, kz))):
        print("\n  --- %s ---" % name)
        hits = []
        for lam, mu in GRID:
            forms, _ = invariant_forms(act, lam, mu, lam, mu, N=N)
            if not forms:
                continue
            for form in forms:
                par, eps, nondeg = form_report(form, N)[:3]
                hits.append((lam, mu, par, eps, nondeg))
                print("    lambda=%-5s mu=%-4s dim=%d  parity=%s  eps=%s  nondeg=%s"
                      % (lam, mu, len(forms), par, sorted(map(str, eps)), nondeg))
        if not hits:
            print("    no invariant form for any (lambda, mu) on the grid")
        out[name] = hits
    return out


def stage2(s, kz, N=5):
    print("\n" + "=" * 72)
    print("STAGE 2.  The duality  cF_{lambda;mu}^# = cF_{lambda*;-mu}")
    print("=" * 72)
    duals = {}
    for name, act in (("NS", act_NS),
                      ("R", lambda l, m, f, u: act_R(l, m, f, u, s, kz))):
        print("\n  --- %s ---" % name)
        for lam, mu in [(Q(l, 2), Q(m, 2)) for l in range(-2, 5) for m in (0, 1)]:
            found = []
            for l2 in [Q(k, 2) for k in range(-6, 9)]:
                for m2 in (-mu, Q(1, 2) - mu):
                    forms, _ = invariant_forms(act, lam, mu, l2, m2, N=N)
                    for form in forms:
                        par, eps, nondeg = form_report(form, N)[:3]
                        found.append((str(l2), str(m2), "par=%s" % par,
                                      "nondeg" if nondeg else "DEG"))
            print("    lambda=%-5s mu=%-4s  pairs with (lambda*, mu*) = %s"
                  % (lam, mu, found))
            if found:
                duals[(name, lam, mu)] = found
    return duals


def build_W(sector, act, lam, mu, N):
    """W = cF_{lambda;mu} (+) cF_{lambda;mu}^#, the pairing computed, not assumed."""
    lam_d = 1 - Q(lam)
    mu_d = -Q(mu) if sector == "NS" else Q(1, 2) - Q(mu)
    forms, _ = invariant_forms(act, lam, mu, lam_d, mu_d, N=N)
    odd = [f for f in forms
           if all((len(ku[1]) + len(kv[1])) % 2 == 1 for (ku, kv) in f)]
    if not odd:
        return None
    pair = odd[0]
    # normalize: make the coefficient at ((0,()), (-1,(1,))) equal to 1 if present
    return Wspace(sector, lam, mu, lam_d, pair, par_shift=1)


def LL(m):
    return mono(m + 1)


def alpha_from_LL(Wsp, bracket, win, ms=(2, 3, 4)):
    """c(L_m, L_{-m}) = alpha m(m^2-1) + kappa m; c = 3 alpha."""
    vals = []
    for m in ms:
        v, rest = cocycle(Wsp, bracket, LL(m), LL(-m), win)
        assert not rest, "not central: %s" % rest
        vals.append((m, v))
    (m1, v1), (m2, v2) = vals[0], vals[1]
    det = m1 ** 3 * m2 - m1 * m2 ** 3
    al = (v1 * m2 - m1 * v2) / det
    ka = (m1 ** 3 * v2 - v1 * m2 ** 3) / det
    for m, v in vals[2:]:
        assert al * m ** 3 + ka * m == v, ("c(L_m,L_-m) is not a cubic", vals)
    return al


def h_of(Wsp, win):
    """h = -(1/8) <hat L_1 hat L_{-1}>, the normal-ordering constant of L_0
    never being needed since [ell_1, ell_{-1}] = -2 ell_0 has no central term."""
    st = hatX(Wsp, LL(1), hatX(Wsp, LL(-1), VAC, win), win)
    return -Q(1, 8) * st.get((), Q(0))


def stage3(s, kz, N=8):
    print("\n" + "=" * 72)
    print("STAGE 3.  Spin(W) for W = cF_{lambda;mu} (+) cF_{lambda;mu}^#")
    print("=" * 72)
    acts = {"NS": act_NS, "R": lambda l, m, f, u: act_R(l, m, f, u, s, kz)}
    brs = {"NS": br_NS, "R": lambda f, g: br_R(f, g, s, kz)}
    print("%-4s %-8s %-8s %-12s %-12s %-10s" %
          ("sec", "lambda", "mu", "c = 3 alpha", "h", "windows"))
    print("-" * 72)
    out = {}
    for sector in ("NS", "R"):
        for lam, mu in ((Q(1, 3), Q(1, 5)), (Q(1, 2), Q(0)), (Q(0), Q(0)),
                        (Q(2), Q(-1)), (Q(-3, 7), Q(5, 2)), (Q(1), Q(1, 3)),
                        (Q(7, 11), Q(-4, 9))):
            W = build_W(sector, acts[sector], lam, mu, N)
            if W is None:
                print("%-4s %-8s %-8s  no odd invariant pairing found" % (sector, lam, mu))
                continue
            res = []
            for win in (4, 5):
                res.append((3 * alpha_from_LL(W, brs[sector], win),
                            h_of(W, win)))
            stable = res[0] == res[1]
            print("%-4s %-8s %-8s %-12s %-12s %-10s" %
                  (sector, lam, mu, res[0][0], res[0][1],
                   "stable" if stable else "UNSTABLE %s" % (res,)))
            out[(sector, lam, mu)] = res[0]
    return out


def stage3b(s, kz, N=8, win=5):
    """the defining property of hat X, checked in both sectors."""
    print("\n  [hat X, psi_v] = psi_{Xv} (both parities), failures:")
    acts = {"NS": act_NS, "R": lambda l, m, f, u: act_R(l, m, f, u, s, kz)}
    for sector in ("NS", "R"):
        W = build_W(sector, acts[sector], Q(1, 3), Q(1, 5), N)
        fs = [mono(a, S) for a in (-1, 0, 1, 2) for S in ((), (1,))]
        print("    %-3s %d" % (sector, test_hat(W, win, fs)))




def fit(points, monoms):
    """exact fit of h (or c) by the given monomials in (lambda, mu); returns
    the coefficient vector if the fit is exact on EVERY point, else None."""
    import sys as _s
    _s.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))
    from exactlin import solve as exsolve, rank as exrank
    A = [[m(l, u) for m in monoms] for (l, u, _v) in points]
    b = [v for (_l, _u, v) in points]
    if exrank(A) < len(monoms):
        return None
    # least-squares by exact normal equations is wrong for over-determined
    # inconsistent systems; instead solve on a square subsystem and verify.
    import itertools as _it
    for rows in _it.combinations(range(len(A)), len(monoms)):
        sub = [A[i] for i in rows]
        if exrank(sub) < len(monoms):
            continue
        x = exsolve(sub, [b[i] for i in rows])
        if x is None:
            continue
        if all(sum(c * m(l, u) for c, m in zip(x, monoms)) == v
               for (l, u, v) in points):
            return x
        return None
    return None


def stage4(s, kz, N=8, win=4):
    print("\n" + "=" * 72)
    print("STAGE 4.  c and h as functions of (lambda, mu) -- exact fit")
    print("=" * 72)
    acts = {"NS": act_NS, "R": lambda l, m, f, u: act_R(l, m, f, u, s, kz)}
    brs = {"NS": br_NS, "R": lambda f, g: br_R(f, g, s, kz)}
    grid = [(Q(l, 3), Q(m, 4)) for l in (-1, 0, 1, 2, 3, 5) for m in (0, 1, 2, 3)]
    monoms = [lambda l, u: Q(1), lambda l, u: l, lambda l, u: u,
              lambda l, u: l * l, lambda l, u: l * u, lambda l, u: u * u]
    names = ["1", "lambda", "mu", "lambda^2", "lambda.mu", "mu^2"]
    for sector in ("NS", "R"):
        cs, hs = [], []
        for lam, mu in grid:
            W = build_W(sector, acts[sector], lam, mu, N)
            if W is None:
                print("  %s (%s, %s): no pairing" % (sector, lam, mu))
                continue
            c = 3 * alpha_from_LL(W, brs[sector], win)
            h = h_of(W, win)
            cs.append((lam, mu, c))
            hs.append((lam, mu, h))
        for what, pts in (("c", cs), ("h", hs)):
            x = fit(pts, monoms)
            if x is None:
                print("  %-3s %s: no exact quadratic fit on %d points" %
                      (sector, what, len(pts)))
                for p in pts:
                    print("        lambda=%-6s mu=%-6s %s=%s" % (p[0], p[1], what, p[2]))
            else:
                terms = " + ".join("%s*%s" % (c, n) for c, n in zip(x, names) if c)
                print("  %-3s %s = %s      (exact on all %d points)" %
                      (sector, what, terms if terms else "0", len(pts)))


if __name__ == "__main__":
    good = stage0()
    if good:
        s, kz = good[0]
        stage0b(s, kz)
        stage1(s, kz)
        stage2(s, kz)
        stage3b(s, kz)
        stage3(s, kz)
        stage4(s, kz)
