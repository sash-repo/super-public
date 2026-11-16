#!/usr/bin/env python3
r"""
runns4.py -- driver for ns4.py.  Answers the questions:

  "In order not to deal with the odd pairing, let us consider the case of 4 odd
   indeterminates, for both NS and Ramond cases, when there is a bilinear form
   on cF_{lambda; mu}; compute the highest weight of Spin and Osci."

STAGE 0  the two algebras and the modules, verified from scratch;
STAGE 1  the degree-0 part g_0 of each algebra, whence the number of the
         parameters of the weighted densities: lambda and mu, and no more;
STAGE 2  which cF_{lambda; mu} carries an invariant bilinear form, and of
         which parity, symmetry and rank;
STAGE 3  the dual module (cF_{lambda; mu})^#;
STAGE 4  Spin and Osc on the modules of STAGE 2: the highest weight
         (c, h; H_1, H_2), computed from the quadratic lift of eq. (150);
STAGE 5  the same for the doubled module W = V (+) V^#, which is what the
         n >= 3 column of the two tables of Subsection (7.4.7) claims to give.
"""

from fractions import Fraction as Q
from itertools import product
import sys
import time

from ns4 import (SUBSETS, SRANK, LPAIRS, MPAIRS, MZETA, MTHETA,
                 TORUS_L, TORUS_M, NAMES_L, NAMES_M,
                 mono, keys, add, smul, mul, parity, clean, d_t, d_odd,
                 cE, nabla, D_M, br_L, br_M, K_L, K_M, act_L, act_M,
                 degL, degM, jacobi_bad, module_bad,
                 invariant_forms, check_invariance, form_report,
                 split_symmetry, Fock, hatX, test_hat, cocycle, psi, VAC)

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "lib"))
from exactlin import rank as exrank, solve as exsolve          # noqa: E402

GENS_L = keys(range(-1, 3))
GENS_M = keys(range(-1, 3))
VERIFY = keys(range(-2, 4))


def nm(k, names):
    s = "".join(names[i] for i in k[1])
    return "t^%s%s" % (k[0], s) if s else "t^%s" % k[0]


def fmt(f, names):
    if not f:
        return "0"
    return " + ".join("%s %s" % (c, nm(k, names)) for k, c in sorted(f.items()))


# ============================================================== STAGE 0

def stage0():
    print("=" * 78)
    print("STAGE 0.  The two algebras and the two families of modules")
    print("=" * 78)
    ks = keys(range(0, 2))                      # 32 monomials
    t0 = time.time()
    print("  k^L(1|4): Jacobi failures on %d^3 monomials: %d   (%.0fs)"
          % (len(ks), jacobi_bad(br_L, ks), time.time() - t0))
    t0 = time.time()
    print("  k^M(1|4): Jacobi failures on %d^3 monomials: %d   (%.0fs)"
          % (len(ks), jacobi_bad(br_M, ks), time.time() - t0))

    print("\n  Module axiom [K_f, K_g].u = K_{f,g}.u:")
    ks, us = keys(range(0, 2)), keys(range(0, 1))
    grid = [(Q(l, 3), Q(m, 4)) for l in (-3, -1, 0, 2) for m in (0, 1, 3)]
    tot = sum(module_bad(act_L, br_L, par, ks, us) for par in grid)
    print("      k^L(1|4): %d (lambda, mu) pairs, %d failures" % (len(grid), tot))
    tot = sum(module_bad(act_M, br_M, par, ks, us) for par in grid)
    print("      k^M(1|4): %d (lambda, mu) pairs, %d failures" % (len(grid), tot))


# ============================================================== STAGE 1

def _span_dim(vecs):
    cols = sorted({k for v in vecs for k in v})
    if not cols:
        return 0
    M = [[v.get(k, Q(0)) for k in cols] for v in vecs]
    return exrank(M)


def stage1():
    print("\n" + "=" * 78)
    print("STAGE 1.  g_0, and the number of the parameters")
    print("=" * 78)
    for name, deg, br, names in (("k^L(1|4)", degL, br_L, NAMES_L),
                                 ("k^M(1|4)", degM, br_M, NAMES_M)):
        g0 = [(Q(a), S) for a in range(-2, 3) for S in SUBSETS
              if deg(a, S) == 0 + 2]        # degree 2 = the functions of g_0
        ev = [k for k in g0 if len(k[1]) % 2 == 0]
        od = [k for k in g0 if len(k[1]) % 2 == 1]
        print("\n  %s_0 is spanned by K_f for f in" % name)
        print("      even: %s" % ", ".join(nm(k, names) for k in ev))
        print("      odd : %s" % ", ".join(nm(k, names) for k in od))
        print("      sdim = (%d|%d)" % (len(ev), len(od)))
        der = []
        for kf, kg in product(g0, repeat=2):
            b = br(mono(*kf), mono(*kg))
            b = {k: v for k, v in b.items() if k in g0}
            if b:
                der.append(b)
        d = _span_dim(der)
        print("      dim [g_0, g_0] = %d, so g_0/[g_0, g_0] has dim %d"
              % (d, len(g0) - d))
        cen = []
        for kf in g0:
            if all(not {k: v for k, v in br(mono(*kf), mono(*kg)).items()}
                   for kg in g0):
                cen.append(kf)
        print("      the center of g_0: %s"
              % (", ".join(nm(k, names) for k in cen) or "0"))


# ============================================================== STAGE 2

LAMS = ([Q(l, 2) for l in range(-8, 5)]
        + [Q(1, 3), Q(-1, 3), Q(-4, 3), Q(-5, 3), Q(-3, 7), Q(-7, 5)])
MUS = [Q(0), Q(1, 4), Q(1, 3), Q(1, 2), Q(2, 3), Q(3, 4), Q(1, 5)]


def sweep(act, torus, gens, lams, mus, N, label):
    hits = []
    for lam in lams:
        for mu in mus:
            forms, idx = invariant_forms(act, (lam, mu), (lam, mu), N=N,
                                         gens=gens, torus=torus)
            if not forms:
                continue
            bad = sum(check_invariance(act, (lam, mu), (lam, mu), f, VERIFY, N)
                      for f in forms)
            sym, asym = split_symmetry(forms)
            info = []
            for kind, fs in (("sym", sym), ("asym", asym)):
                for f in fs:
                    par, eps, nd, _ = form_report(f, N)
                    info.append((kind, par, nd))
            print("    %s lambda=%-7s mu=%-5s dim=%d  %s   (invariance defects: %d)"
                  % (label, lam, mu, len(forms),
                     "; ".join("%s, parity %s, %s" % (k, p, "nondeg" if n
                                                      else "DEGENERATE")
                               for k, p, n in info), bad))
            hits.append((lam, mu, forms, sym, asym))
    if not hits:
        print("    %s: no invariant form anywhere on the grid" % label)
    return hits


def stage2(N=4):
    print("\n" + "=" * 78)
    print("STAGE 2.  The modules cF_{lambda; mu} carrying an invariant form")
    print("=" * 78)
    print("  grid: %d values of lambda x %d values of mu, window |a| <= %d"
          % (len(LAMS), len(MUS), N))
    t0 = time.time()
    hL = sweep(act_L, TORUS_L, GENS_L, LAMS, MUS, N, "L")
    print("  (k^L: %.0fs)" % (time.time() - t0))
    t0 = time.time()
    hM = sweep(act_M, TORUS_M, GENS_M, LAMS, MUS, N, "M")
    print("  (k^M: %.0fs)" % (time.time() - t0))
    return hL, hM


# ============================================================== STAGE 3

def stage3(N=4):
    print("\n" + "=" * 78)
    print("STAGE 3.  The dual module")
    print("=" * 78)
    for label, act, torus, gens, shifts in (
            ("L", act_L, TORUS_L, GENS_L, (Q(0),)),
            ("M", act_M, TORUS_M, GENS_M, (Q(0), Q(1, 2), Q(-1, 2)))):
        for lam, mu in ((Q(1, 3), Q(1, 5)), (Q(0), Q(0)), (Q(2), Q(1, 2)),
                        (Q(-1), Q(1, 4))):
            found = []
            for l2 in [Q(k, 2) for k in range(-10, 7)]:
                for sh in shifts:
                    m2 = sh - mu
                    forms, _ = invariant_forms(act, (lam, mu), (l2, m2), N=N,
                                               gens=gens, torus=torus)
                    for f in forms:
                        par, eps, nd, _ = form_report(f, N)
                        found.append("(%s; %s) parity %s %s"
                                     % (l2, m2, par,
                                        "nondeg" if nd else "DEG"))
            print("    %s  (%s; %s)^#  =  %s" % (label, lam, mu,
                                                 ", ".join(found) or "not found"))


# ================================================== building the Fock spaces

def pair_const(form):
    """the constant a + a' of the pairing; the window must be symmetric on it."""
    sums = {ku[0] + kv[0] for (ku, kv), c in form.items() if c}
    if len(sums) != 1:
        raise ValueError("the pairing has no constant t-degree: %s"
                         % sorted(map(str, sums)))
    return Q(sums.pop())


def sigma_closed(K, N):
    """the largest sigma-closed range of t-degrees inside [-N, N]:
    a and K - a both in the window."""
    lo, hi = max(-N, K - N), min(N, K + N)
    return [Q(a) for a in range(int(lo), int(hi) + 1)]


def build_V(act, par, form, N, shift):
    """Spin (shift = 1) or Osc (shift = 0) on the module itself."""
    K = pair_const(form)
    modes = [(a, S) for a in sigma_closed(K, N) for S in SUBSETS]

    def Bfun(m, n):
        return form.get((m, n), Q(0))

    def actfun(f, m):
        return act(par, f, {m: Q(1)})

    return Fock(modes, lambda m: len(m[1]) % 2, Bfun, actfun,
                lambda m: (m[0], SRANK[m[1]]), shift)


def build_W(act, par, par_d, form, N, shift, par_shift=0):
    """Spin/Osc on the doubled module W = V (+) V^#; modes are (s, a, S)."""
    K = pair_const(form)
    lo, hi = max(-N, K - N), min(N, K + N)
    arange = [Q(a) for a in range(int(lo), int(hi) + 1)]
    modes = [(s, a, S) for s in (0, 1) for a in arange for S in SUBSETS]
    sig = {}
    for (ku, kv), c in form.items():
        if c:
            sig[(0,) + ku] = (1,) + kv
            sig[(1,) + kv] = (0,) + ku

    def pfun(m):
        return (len(m[2]) + (par_shift if m[0] == 1 else 0)) % 2

    def Bfun(m1, m2):
        (s1, a1, S1), (s2, a2, S2) = m1, m2
        if s1 == s2:
            return Q(0)
        if s1 == 0:
            return form.get(((a1, S1), (a2, S2)), Q(0))
        return (-1) ** (pfun(m1) * pfun(m2)) * \
            form.get(((a2, S2), (a1, S1)), Q(0))

    def actfun(f, m):
        s, a, S = m
        p = par if s == 0 else par_d
        r = act(p, f, {(a, S): Q(1)})
        return {(s, k[0], k[1]): v for k, v in r.items()}

    return Fock(modes, pfun, Bfun, actfun,
                lambda m: (m[1], m[0], SRANK[m[2]]), shift)


# =============================================== the labels of the highest weight

def LL(m):
    return mono(m + 1)


def charge(W, bracket, ms=(2, 3, 4)):
    """c from c(K_{t^{m+1}}, K_{t^{1-m}}) = alpha m^3 + kappa m, c = 3 alpha."""
    vals = []
    for m in ms:
        v, rest = cocycle(W, bracket, LL(m), LL(-m))
        if rest:
            raise ValueError("the 2-cocycle is not central: %s" % rest)
        vals.append((m, v))
    (m1, v1), (m2, v2) = vals[0], vals[1]
    det = m1 ** 3 * m2 - m1 * m2 ** 3
    al = (v1 * m2 - m1 * v2) / det
    ka = (m1 ** 3 * v2 - v1 * m2 ** 3) / det
    for m, v in vals[2:]:
        if al * m ** 3 + ka * m != v:
            raise ValueError("c(L_m, L_-m) is not a cubic: %s" % (vals,))
    return 3 * al


def h_direct(W):
    """h = the eigenvalue of ell_0 = (1/2) hat K_t on the vacuum."""
    st = hatX(W, mono(1), VAC)
    rest = {k: v for k, v in st.items() if k != ()}
    if rest:
        raise ValueError("hat K_t does not fix the vacuum: %s" % rest)
    return Q(1, 2) * st.get((), Q(0))


def h_via_L1(W):
    """h = -(1/8) <hat K_{t^2} hat K_1>, since [ell_1, ell_{-1}] = -2 ell_0."""
    st = hatX(W, LL(1), hatX(W, LL(-1), VAC))
    return -Q(1, 8) * st.get((), Q(0))


def H_of(W, torus):
    out = []
    for g in torus:
        st = hatX(W, mono(*g), VAC)
        rest = {k: v for k, v in st.items() if k != ()}
        if rest:
            raise ValueError("hat K_{%s} does not fix the vacuum: %s"
                             % (g, rest))
        out.append(st.get((), Q(0)))
    return tuple(out)


def labels(W, bracket, torus):
    c = charge(W, bracket)
    h1, h2 = h_direct(W), h_via_L1(W)
    if h1 != h2:
        raise ValueError("the two computations of h disagree: %s vs %s"
                         % (h1, h2))
    return c, h1, H_of(W, torus)


# ============================================================== STAGE 4

def stage4(N=4, wins=(3, 4)):
    print("\n" + "=" * 78)
    print("STAGE 4.  Spin and Osc on the modules that carry a form of their own")
    print("=" * 78)
    for label, act, br, torus, gens in (("L", act_L, br_L, TORUS_L, GENS_L),
                                        ("M", act_M, br_M, TORUS_M, GENS_M)):
        print("\n  --- k^%s(1|4) ---" % label)
        hits = sweep(act, torus, gens, LAMS, MUS, N, label)
        for lam, mu, forms, sym, asym in hits:
            for kind, fs, shift in (("Spin", sym, 1), ("Osc", asym, 0)):
                for form in fs:
                    row = []
                    for W in (build_V(act, (lam, mu), form, w, shift)
                              for w in wins):
                        try:
                            row.append(labels(W, br, torus))
                        except ValueError as e:
                            row.append("FAILED: %s" % e)
                    print("      %s(cF_{%s; %s}):  %s   %s"
                          % (kind, lam, mu, row[0],
                             "stable" if row[0] == row[-1]
                             else "UNSTABLE %s" % (row,)))


# ============================================================== STAGE 5

def DUAL_L(par):
    lam, mu = par
    return (-2 - lam, -mu)


def DUAL_M(par):
    lam, mu = par
    return (-2 - lam, Q(1, 2) - mu)


def stage5(N=4, wins=(3, 4)):
    print("\n" + "=" * 78)
    print("STAGE 5.  Spin and Osc on the doubled module W = V (+) V^#")
    print("=" * 78)
    for label, act, br, torus, gens, rule, grid in (
            ("L", act_L, br_L, TORUS_L, GENS_L, DUAL_L,
             [(Q(l, 3), Q(m, 4)) for l in (-4, -3, -1, 0, 2)
              for m in (0, 1, 3)] + [(Q(-3, 7), Q(2, 9))]),
            ("M", act_M, br_M, TORUS_M, GENS_M, DUAL_M,
             [(Q(l, 3), Q(m, 4)) for l in (-4, -3, -1, 0, 2)
              for m in (0, 1, 3)] + [(Q(-3, 7), Q(2, 9))])):
        print("\n  --- k^%s(1|4) ---" % label)
        for par in grid:
            par_d = rule(par)
            forms, _ = invariant_forms(act, par, par_d, N=N, gens=gens,
                                       torus=torus)
            if not forms:
                print("      (%s; %s): no invariant pairing with (%s; %s)"
                      % (par[0], par[1], par_d[0], par_d[1]))
                continue
            form = forms[0]
            for kind, shift in (("Spin", 1), ("Osc", 0)):
                row = []
                for w in wins:
                    W = build_W(act, par, par_d, form, w, shift)
                    try:
                        row.append(labels(W, br, torus))
                    except ValueError as e:
                        row.append("FAILED: %s" % e)
                print("      %s(cF_{%s; %s} (+) dual): %s  %s"
                      % (kind, par[0], par[1], row[0],
                         "stable" if row[0] == row[-1]
                         else "UNSTABLE %s" % (row,)))


if __name__ == "__main__":
    which = sys.argv[1:] or ["0", "1", "2", "3", "4"]
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
