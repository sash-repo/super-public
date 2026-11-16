#!/usr/bin/env python3
r"""
runvect2.py -- driver for vect2.py.  Answers the questions:

  "Now, please 'do the same' = compute highest weight for Spin(W) and Spin(V)
   for \fvect^L(1|2) and next \fsvect_lambda^L(1|2)"

STAGE 0  the conventions: bracket, divergence, the basis eq. equation 77 of
         the text, the modules -- all verified from scratch;
STAGE 1  the degree-zero components g_0, whence the number of parameters of
         the rank-1 modules;
STAGE 2  which modules carry an invariant form, and what the dual module is;
STAGE 3  Spin(W) for W = V (+) V^#: c, h and the labels H;
STAGE 4  the Spin cocycle against eq. equation 76 and equation 78;
STAGE 5  Spin(V) undoubled, where V is self-dual;
STAGE 6  h in closed form: the second difference of the Bernoulli B_2 of the
         fractional parts, verified on a grid.
"""

from fractions import Fraction as Q
from itertools import product
import sys

from vect2 import (VF, VARS, vadd, vsmul, vzero, vparity, vname, apply_vf,
                   br_vf, div, vect_basis, deg_vf, ell, Sfield, svect_basis,
                   svect_kernel, act, modpar, module_bad, jacobi_bad, weight,
                   invariant_forms, form_report, check_invariance, PairRule,
                   Wspace, zero_couples, test_hat, cw_relations_bad, cocycle,
                   cubic, hat_eigen, labels)
from ns2 import mono, add, smul, mul, keys, SUBSETS

N_FORM = 5                                   # window in which forms are solved
WIN = 5                                      # window of the Fock computations


def sgens(lam, M=2):
    """generators of svect^L_lambda(1|2) in the doubled degrees |d| <= 2M+1."""
    out = []
    for m in range(-M, M + 1):
        ev, od = svect_basis(m, lam)
        out += [D for _, D in ev] + [D for _, D in od]
    return out


def vgens(M=2):
    return vect_basis(M)


def frac(x):
    return x - (x.numerator // x.denominator)


def B2(x):
    return x * x - x + Q(1, 6)


def h_formula(mu_eff, lam):
    """h = nu^2 - (1/2)[B_2{x_0} - 2 B_2{x_1} + B_2{x_2}], nu = (lam+1)/2."""
    nu = Q(lam + 1, 1) / 2
    xs = [mu_eff + j * nu for j in (0, 1, 2)]
    return nu * nu - Q(1, 2) * (B2(frac(xs[0])) - 2 * B2(frac(xs[1]))
                                + B2(frac(xs[2])))


# ================================================================== STAGE 0

def stage0():
    print("=" * 78)
    print("STAGE 0.  The conventions, verified from scratch")
    print("=" * 78)
    fs = vect_basis(1)
    print("  vect^L(1|2): %d monomial fields with |a| <= 1" % len(fs))
    print("  Jacobi failures on %d^3 fields: %d" % (18, jacobi_bad(fs[:18])))
    bad = 0
    for D, E in product(fs, repeat=2):
        s = (-1) ** (vparity(D) * vparity(E))
        if add(div(br_vf(D, E)),
               smul(-1, add(apply_vf(D, div(E)), smul(-s, apply_vf(E, div(D)))))):
            bad += 1
    print("  Div [D, E] = D(Div E) -+ E(Div D): %d failures of %d pairs"
          % (bad, len(fs) ** 2))

    for lam in (Q(0), Q(1, 3), Q(2, 5)):
        bad = 0
        for m in range(-3, 4):
            for n in range(-3, 4):
                if not vzero(vadd(br_vf(ell(m, lam), ell(n, lam)),
                                  vsmul(-(n - m), ell(m + n, lam)))):
                    bad += 1
        print("  lambda = %-5s [LL_m, LL_n] = (n-m) LL_{m+n}: %d failures of 49; "
              "Div LL_m = %s t^m" % (lam, bad, div(ell(1, lam)).get((Q(1), ()), 0)))

    print("\n  the basis eq. equation 77: every element satisfies "
          "lambda f_t = -t Div D, and it spans the whole kernel")
    for lam in (Q(0), Q(1, 3), Q(2)):
        for m in (-1, 0, 1):
            ev, od = svect_basis(m, lam)
            for pw, named in ((0, ev), (1, od)):
                ker, _ = svect_kernel(m, lam, pw)
                out = [nm for nm, D in named
                       if add(smul(lam, D.get("t", {})), mul(mono(1), div(D)))]
                print("    lambda=%-5s degree %2d: |basis| = %d, dim ker = %d, "
                      "outside: %s" % (lam, 2 * m + pw, len(named), len(ker),
                                       out or "none"))

    print("\n  the module axiom [L_D, L_E] = L_{[D,E]} on Vol^a_mu:")
    us = keys(range(-1, 2))
    for par in ((Q(0), Q(0)), (Q(1, 2), Q(0)), (Q(1, 3), Q(1, 5)),
                (Q(-2), Q(2, 7))):
        print("    Vol^%s_%s: %d failures of %d"
              % (par[0], par[1], module_bad(par, fs[:12], us), 12 * 12 * len(us)))

    print("\n  over svect^L_lambda(1|2): Vol^a_mu = cF_{0; mu - a lambda}")
    for lam in (Q(1, 3), Q(2, 5)):
        for a, mu in ((Q(1, 2), Q(1, 7)), (Q(2), Q(0)), (Q(-1, 3), Q(1, 5))):
            bad = 0
            for D in sgens(lam, 1):
                for ku in us:
                    u = {ku: Q(1)}
                    if add(act((a, mu), D, u),
                           smul(-1, act((Q(0), mu - a * lam), D, u))):
                        bad += 1
            print("    lambda=%-5s a=%-5s mu=%-5s: %d failures" % (lam, a, mu, bad))


# ================================================================== STAGE 1

def stage1():
    print("\n" + "=" * 78)
    print("STAGE 1.  The degree-zero components g_0")
    print("=" * 78)
    ev = [D for D in vect_basis(2) if deg_vf(D) == 0 and vparity(D) == 0]
    od = [D for D in vect_basis(2) if deg_vf(D) == 0 and vparity(D) == 1]
    print("  vect^L(1|2)_0 = <%s>" % ", ".join(vname(D) for D in ev))
    print("      sdim = (%d|%d)  (the odd part of g sits in the odd degrees)"
          % (len(ev), len(od)))
    Z = VF(ft=mono(0, (1, 2)))
    E = VF(f1=mono(0, (1,)), f2=mono(0, (2,)))
    L0 = ell(0)
    for nm, D in (("t d_t", VF(ft=mono(1))), ("cE", E), ("ell_0", L0),
                  ("w1 d_2", VF(f2=mono(0, (1,)))),
                  ("w1 d_1 - w2 d_2",
                   VF(f1=mono(0, (1,)), f2=smul(-1, mono(0, (2,)))))):
        print("      [%s, Z] = %s" % (nm, vname(br_vf(D, Z))))
    print("      [Z, Z] = %s;  so g_0 = C ell_0 (+) sl(2) (+) <cE, Z>, "
          "the center being C ell_0" % vname(br_vf(Z, Z)))
    for lam in (Q(0), Q(1, 3), Q(2, 5)):
        ker, _ = svect_kernel(0, lam, 0)
        print("  svect^L_%s(1|2)_0: sdim = (%d|0),  basis %s"
              % (lam, len(ker), ", ".join(vname(D) for D in ker)))


# ================================================================== STAGE 2

def stage2():
    print("\n" + "=" * 78)
    print("STAGE 2.  The invariant forms and the dual module")
    print("=" * 78)
    print("  over vect^L(1|2): (Vol^a_mu)^# = ?")
    a, mu = Q(1, 3), Q(1, 7)
    for cand in ((1 - a, -mu), (1 - a, mu), (-a, -mu), (a, -mu), (2 - a, -mu),
                 (1 - a, -mu + 1)):
        forms, idx = invariant_forms((a, mu), cand, vgens(), N=N_FORM)
        tag = ""
        if forms:
            R = PairRule(forms[0])
            p, eps, nondeg, _ = form_report(forms[0], N_FORM)
            tag = ("parity %d, eps = %s, non-degenerate %s, K = %s, "
                   "invariance failures %d of %d"
                   % ((p, sorted(map(str, eps)), nondeg, R.K)
                      + check_invariance(forms[0], (a, mu), cand, vgens(),
                                         N_FORM)))
        print("    V = Vol^%s_%s, candidate Vol^%s_%s: dim = %d  %s"
              % (a, mu, cand[0], cand[1], len(forms), tag))
    print("  self-duality over vect^L(1|2): Vol^a_mu = (Vol^a_mu)^# iff "
          "a = 1/2 and 2 mu in Z")
    for (a, mu) in ((Q(1, 2), Q(0)), (Q(1, 2), Q(1, 2)), (Q(1, 2), Q(1)),
                    (Q(1, 2), Q(1, 3)), (Q(1, 3), Q(0)), (Q(0), Q(0))):
        forms, _ = invariant_forms((a, mu), (a, mu), vgens(), N=N_FORM)
        print("    a = %-5s mu = %-5s: dim = %d %s"
              % (a, mu, len(forms),
                 form_report(forms[0], N_FORM)[:2] if forms else ""))
    print("\n  over svect^L_lambda(1|2): (cF_mu)^# = cF_{-lambda-mu}")
    for lam in (Q(1, 3), Q(2, 5)):
        for cand in (-lam - Q(1, 7), -lam - Q(1, 7) + 1, -Q(1, 7), Q(1, 7)):
            forms, _ = invariant_forms((Q(0), Q(1, 7)), (Q(0), cand),
                                       sgens(lam), N=N_FORM, lam=lam)
            print("    lambda = %-5s mu = 1/7, candidate %-8s: dim = %d %s"
                  % (lam, cand, len(forms),
                     form_report(forms[0], N_FORM)[:2] if forms else ""))
    print("  self-duality over svect^L_lambda(1|2): iff 2 mu + lambda in Z")
    for lam in (Q(1, 3), Q(2, 5)):
        for mu in (-lam / 2, -lam / 2 + Q(1, 2), -lam / 2 + 1, Q(1, 7)):
            forms, _ = invariant_forms((Q(0), mu), (Q(0), mu), sgens(lam),
                                       N=N_FORM, lam=lam)
            print("    lambda = %-5s mu = %-8s: dim = %d %s"
                  % (lam, mu, len(forms),
                     form_report(forms[0], N_FORM)[:2] if forms else ""))


# ============================================================ building W and V

def W_vect(a, mu):
    forms, _ = invariant_forms((a, mu), (1 - a, -mu), vgens(), N=N_FORM)
    if len(forms) != 1:
        return None
    return Wspace((a, mu), (1 - a, -mu), PairRule(forms[0]))


def W_svect(lam, mu):
    forms, _ = invariant_forms((Q(0), mu), (Q(0), -lam - mu), sgens(lam),
                               N=N_FORM, lam=lam)
    if len(forms) != 1:
        return None
    return Wspace((Q(0), mu), (Q(0), -lam - mu), PairRule(forms[0]), lam=lam)


def V_vect(a, mu, zc=()):
    forms, _ = invariant_forms((a, mu), (a, mu), vgens(), N=N_FORM)
    if len(forms) != 1:
        return None
    return Wspace((a, mu), (a, mu), PairRule(forms[0]), undoubled=True,
                  zero_creation=zc)


def V_svect(lam, mu, zc=()):
    forms, _ = invariant_forms((Q(0), mu), (Q(0), mu), sgens(lam),
                               N=N_FORM, lam=lam)
    if len(forms) != 1:
        return None
    return Wspace((Q(0), mu), (Q(0), mu), PairRule(forms[0]), lam=lam,
                  undoubled=True, zero_creation=zc)


G0V = [("t d_t", VF(ft=mono(1))),
       ("w1 d_1", VF(f1=mono(0, (1,)))),
       ("w2 d_2", VF(f2=mono(0, (2,)))),
       ("w1 d_2", VF(f2=mono(0, (1,)))),
       ("w2 d_1", VF(f1=mono(0, (2,)))),
       ("w1w2 d_t", VF(ft=mono(0, (1, 2))))]
HSV = VF(f1=mono(0, (1,)), f2=smul(-1, mono(0, (2,))))


def g0_values(W, win, names=G0V):
    out = []
    for nm, D in names:
        try:
            out.append((nm, hat_eigen(W, D, win)))
        except ValueError:
            out.append((nm, "NOT A SCALAR"))
    return out


# ================================================================== STAGE 3

def stage3():
    print("\n" + "=" * 78)
    print("STAGE 3.  Spin(W), W = V (+) V^#: the highest weight")
    print("=" * 78)
    W = W_vect(Q(1, 3), Q(1, 7))
    print("  the Clifford-Weyl relations on W: %d failures of %d"
          % cw_relations_bad(W, 3))
    print("  [hat X, psi_v] = psi_{Xv} for even X: %d failures of %d"
          % test_hat(W, WIN, [D for D in vect_basis(1) if vparity(D) == 0][:12]))

    print("\n  vect^L(1|2), V = Vol^a_mu:")
    print("  %-7s %-7s %-6s %-7s %-7s %-7s %-7s %-7s %s"
          % ("a", "mu", "c", "kappa", "h", "h(comm)", "H_1", "H_2", "window"))
    for a in (Q(1, 3), Q(0), Q(1, 2), Q(1), Q(-2, 5), Q(7, 3)):
        for mu in (Q(1, 7), Q(-1, 7), Q(1, 3), Q(2, 3), Q(1, 5), Q(-3, 5),
                   Q(2, 9), Q(9, 20), Q(11, 20), Q(1, 20), Q(19, 20)):
            W = W_vect(a, mu)
            if W is None or W.zero_modes(WIN + 1):
                print("  %-7s %-7s  zero modes: %d"
                      % (a, mu, len(W.zero_modes(WIN + 1))))
                continue
            c, h, hc, ka, cok = labels(W, WIN)
            c2 = labels(W, WIN + 1)
            vals = dict(g0_values(W, WIN))
            print("  %-7s %-7s %-6s %-7s %-7s %-7s %-7s %-7s %s"
                  % (a, mu, c, ka, h, hc, vals["w1 d_1"], vals["w2 d_2"],
                     ("stable" if c2[:3] == (c, h, hc) else "UNSTABLE %s" % (c2,)) + ("" if cok else " (m=4 off-window)")))
    W = W_vect(Q(1, 3), Q(1, 7))
    print("  the other elements of g_0 on the vacuum: %s"
          % ", ".join("%s = %s" % kv for kv in g0_values(W, WIN)))

    print("\n  svect^L_lambda(1|2), V = cF_mu:")
    print("  %-8s %-8s %-6s %-9s %-8s %-8s %-4s %s"
          % ("lambda", "mu", "c", "kappa", "h", "h(comm)", "H", "window"))
    for lam in (Q(1, 3), Q(2, 5), Q(-3, 7), Q(1, 2), Q(3, 5), Q(-2, 3), Q(5, 4)):
        for mu in (Q(1, 7), Q(2, 9), Q(1, 5), Q(3, 7)):
            W = W_svect(lam, mu)
            if W is None or W.zero_modes(WIN + 1):
                print("  %-8s %-8s  zero modes" % (lam, mu))
                continue
            c, h, hc, ka, cok = labels(W, WIN, lam)
            c2 = labels(W, WIN + 1, lam)
            H = hat_eigen(W, HSV, WIN)
            print("  %-8s %-8s %-6s %-9s %-8s %-8s %-4s %s"
                  % (lam, mu, c, ka, h, hc, H,
                     ("stable" if c2[:3] == (c, h, hc) else "UNSTABLE %s" % (c2,)) + ("" if cok else " (m=4 off-window)")))


# ================================================================== STAGE 4

def stage4():
    print("\n" + "=" * 78)
    print("STAGE 4.  The Spin cocycle against eqs. equation 76 and equation 78")
    print("=" * 78)

    def Efield(k, i, j):
        return VF(**{("f1" if j == 1 else "f2"): mono(k, (i,))})

    W = W_vect(Q(1, 3), Q(1, 7))
    print("  over vect^L(1|2) [a = 1/3, mu = 1/7], eq. equation 76:")
    bad = 0
    for k in range(-3, 4):
        for l in range(-3, 4):
            v, rest = cocycle(W, Efield(k, 1, 1), Efield(l, 2, 2), WIN)
            want = Q(k) if l == -k else Q(0)
            bad += (v != want) or bool(rest)
            v, rest = cocycle(W, Efield(k, 1, 2), Efield(l, 2, 1), WIN)
            want = Q(-k) if l == -k else Q(0)
            bad += (v != want) or bool(rest)
            v, rest = cocycle(W, Efield(k, 1, 1), Efield(l, 1, 1), WIN)
            bad += (v != 0) or bool(rest)
            v, rest = cocycle(W, VF(ft=mono(k + 1)), VF(ft=mono(l + 1)), WIN)
            bad += (v != 0) or bool(rest)
    print("    c(t^k w1 d_1, t^l w2 d_2) = k delta, "
          "c(t^k w1 d_2, t^l w2 d_1) = -k delta,")
    print("    c(t^k w1 d_1, t^l w1 d_1) = c(t^{k+1} d_t, t^{l+1} d_t) = 0:")
    print("      %d defects over 4 x 49 pairs -- eq. equation 76 reproduced "
          "with coefficient 1" % bad)

    print("\n  over svect^L_lambda(1|2), eq. equation 78, even entries:")
    print("  %-8s %-4s %-16s %-16s %-16s"
          % ("lambda", "m", "c(LL_m, LL_-m)", "printed", "ratio"))
    for lam in (Q(0), Q(1, 3), Q(2, 5), Q(-3, 7)):
        W = W_svect(lam, Q(1, 7)) if lam else W_vect(Q(0), Q(1, 7))
        for m in (2, 3):
            v, rest = cocycle(W, ell(m, lam), ell(-m, lam), WIN)
            pr = Q(1, 2) * m * (m * m - (lam + 1) ** 2)
            print("  %-8s %-4d %-16s %-16s %-16s"
                  % (lam, m, v, pr, v / pr if pr else "-"))
        for m in (1, 2, 3):
            v, _ = cocycle(W, VF(f1=mono(m, (1,)), f2=smul(-1, mono(m, (2,)))),
                           VF(f1=mono(-m, (1,)), f2=smul(-1, mono(-m, (2,)))),
                           WIN)
            v2, _ = cocycle(W, VF(f2=mono(m, (1,))), VF(f1=mono(-m, (2,))), WIN)
            print("  %-8s %-4s c(t^m(w1d_1-w2d_2), .) = %-8s [printed %s];  "
                  "c(t^m w1 d_2, t^-m w2 d_1) = %-6s [printed %s]"
                  % (lam, m, v, 2 * m, v2, m))


# ================================================================== STAGE 5

def stage5():
    print("\n" + "=" * 78)
    print("STAGE 5.  Spin(V) undoubled, V self-dual")
    print("=" * 78)
    print("  vect^L(1|2): V = sqrt(Vol) (mu in Z) and sqrt(t Vol) (mu in 1/2+Z)")
    for mu in (Q(0), Q(1), Q(-1), Q(1, 2), Q(-1, 2), Q(3, 2)):
        W0 = V_vect(Q(1, 2), mu)
        cps = zero_couples(W0, WIN + 1)
        print("    mu = %-6s K = %-3s zero couples: %s"
              % (mu, W0.pair.K, [(m[1:], s[1:]) for m, s in cps] or "none"))
        for pick in (0, 1):
            W = V_vect(Q(1, 2), mu, zc=[cp[pick] for cp in cps])
            c, h, hc, ka, cok = labels(W, WIN)
            c2 = labels(W, WIN + 1)
            vals = dict(g0_values(W, WIN))
            print("      creation = %-22s c = %-5s kappa = %-5s h = %-6s "
                  "H_1 = %-5s H_2 = %-5s %s"
                  % (str([cp[pick][1:] for cp in cps]), c, ka, h,
                     vals["w1 d_1"], vals["w2 d_2"],
                     ("stable" if c2[:3] == (c, h, hc) else "UNSTABLE") + ("" if cok else " (m=4 off-window)")))
    print("\n  svect^L_lambda(1|2): V = cF_mu with 2 mu + lambda in Z")
    print("  %-8s %-9s %-6s %-9s %-8s %-5s %-24s %s"
          % ("lambda", "mu", "c", "kappa", "h", "H", "doubled (c, kappa, h)",
             "half?"))
    for lam in (Q(1, 3), Q(2, 5), Q(-3, 7), Q(1, 2), Q(3, 5), Q(-2, 3)):
        for mu in (-lam / 2, -lam / 2 + 1, -lam / 2 + Q(1, 2)):
            V = V_svect(lam, mu)
            if V is None:
                print("  %-8s %-9s no invariant form" % (lam, mu))
                continue
            cps = zero_couples(V, WIN + 1)
            if cps:
                print("  %-8s %-9s zero couples %s -- polarization ambiguous"
                      % (lam, mu, [(m[1:], s[1:]) for m, s in cps]))
                continue
            c, h, hc, ka, cok = labels(V, WIN, lam)
            c2 = labels(V, WIN + 1, lam)
            H = hat_eigen(V, HSV, WIN)
            W = W_svect(lam, mu)
            dbl = (labels(W, WIN, lam)
                   if W and not W.zero_modes(WIN + 1) else None)
            print("  %-8s %-9s %-6s %-9s %-8s %-5s %-24s %s"
                  % (lam, mu, c, ka, h, H,
                     "(%s, %s, %s)" % (dbl[0], dbl[3], dbl[1]) if dbl else "-",
                     ("yes" if dbl and (2 * c, 2 * ka, 2 * h)
                      == (dbl[0], dbl[3], dbl[1]) else "NO") +
                     ("" if c2[:3] == (c, h, hc) else " UNSTABLE")))


# ================================================================== STAGE 6

def stage6():
    print("\n" + "=" * 78)
    print("STAGE 6.  h in closed form")
    print("=" * 78)
    print("  h = nu^2 - (1/2)[B_2{mu} - 2 B_2{mu+nu} + B_2{mu+2nu}], "
          "nu = (lambda+1)/2")
    bad = tot = 0
    print("  vect^L(1|2):")
    for a in (Q(0), Q(1, 3), Q(1, 2), Q(1), Q(-2, 5)):
        for num in range(1, 20):
            mu = Q(num, 20)
            if 2 * mu == 1:
                continue
            W = W_vect(a, mu)
            if W.zero_modes(WIN + 1):
                continue
            tot += 1
            h = labels(W, WIN)[1]
            if h != h_formula(mu, Q(0)):
                bad += 1
                print("    MISMATCH a=%s mu=%s: %s vs %s"
                      % (a, mu, h, h_formula(mu, Q(0))))
    print("    %d values, %d mismatches" % (tot, bad))
    bad = tot = 0
    print("  svect^L_lambda(1|2):")
    for lam in (Q(1, 3), Q(2, 5), Q(-3, 7), Q(1, 2), Q(3, 5), Q(-2, 3),
                Q(5, 4), Q(-7, 5)):
        for num in (1, 2, 3, 4, 6, 7, 8, 9, 11, 13):
            mu = Q(num, 14)
            W = W_svect(lam, mu)
            if W is None or W.zero_modes(WIN + 1):
                continue
            tot += 1
            h = labels(W, WIN, lam)[1]
            if h != h_formula(mu, lam):
                bad += 1
                print("    MISMATCH lambda=%s mu=%s: %s vs %s"
                      % (lam, mu, h, h_formula(mu, lam)))
    print("    %d values, %d mismatches" % (tot, bad))
    print("\n  the same for the modules Vol^a_mu over svect (mu_eff = mu - a lambda):")
    bad = tot = 0
    for lam in (Q(1, 3), Q(2, 5)):
        for a in (Q(1, 2), Q(2), Q(-1, 3)):
            for num in (1, 3, 5, 9, 11):
                mu = Q(num, 14)
                W = W_svect(lam, mu - a * lam)
                if W is None or W.zero_modes(WIN + 1):
                    continue
                tot += 1
                if labels(W, WIN, lam)[1] != h_formula(mu - a * lam, lam):
                    bad += 1
    print("    %d values, %d mismatches" % (tot, bad))


if __name__ == "__main__":
    which = sys.argv[1:] or ["0", "1", "2", "3", "4", "5", "6"]
    for s in which:
        globals()["stage" + s]()
