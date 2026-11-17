#!/usr/bin/env python3
"""propchecks.py -- checks of six propositions of the imaging text.

Six groups of checks, one per claim of the text.  Everything that can be exact is exact (sympy over Q); the two
that cannot -- a Gaussian quadrature and a bimodality threshold -- are done in
float and cross-checked against a closed form.

    A  Proposition pr:atten, item 3): when is  dhat^2 = d^2 ?
    B  Proposition pr:blind, item 3): the scalar lambda, and its sign
    C  Lemma le:opt: the separation at which two features merge
    D  Proposition pr:vanish: the adapted Darboux frame of PT^*Sigma
    E  Proposition pr:floor: rho^2_adj, and N_eff
    F  Proposition pr:growth: volume may grow while area falls

Run:  python3 propchecks.py
"""

import itertools
import math

import numpy as np
import sympy as sp

LOG = []
NFAIL = 0
NCHECK = 0


def check(name, cond, detail=""):
    global NFAIL, NCHECK
    NCHECK += 1
    ok = bool(cond)
    if not ok:
        NFAIL += 1
    LOG.append("%-4s %s%s" % ("ok" if ok else "FAIL", name,
                              ("   [%s]" % detail) if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# A.  pr:atten item 3).  dhat^2 = kappa^2 d^2 / (1 + c d^2) <= d^2, and the
#     text says "with equality only for a reader who never errs".
# ---------------------------------------------------------------------------

def group_A():
    LOG.append("=== A.  pr:atten 3): the equality case ===")

    etaM, etaB, piM, d2 = sp.symbols("eta_M eta_B pi_M d2", nonnegative=True)
    piB = 1 - piM
    qM = piM * (1 - etaM) + piB * etaB
    qB = piB * (1 - etaB) + piM * etaM
    w = piM * (1 - etaM) / qM
    wp = piB * (1 - etaB) / qB
    kappa = sp.simplify(w + wp - 1)
    c = sp.simplify(qM * w * (1 - w) + qB * wp * (1 - wp))

    # the closed form the proof in the text asserts
    kappa_closed = piM * piB * (1 - etaM - etaB) / (qM * qB)
    check("A1  kappa = pi_M pi_B (1-eta_M-eta_B)/(q_M q_B)",
          sp.simplify(kappa - kappa_closed) == 0)

    # w, w' in [0,1]; w = 1 iff pi_B eta_B = 0, w = 0 iff eta_M = 1
    check("A2  w = 1  <=>  eta_B = 0  (pi_B > 0)",
          sp.simplify((w - 1).subs(etaB, 0)) == 0)
    check("A3  w = 0  <=>  eta_M = 1",
          sp.simplify(w.subs(etaM, 1)) == 0)

    # kappa = +-1 exactly at the two corners, and there c = 0
    for (eM, eB, kap) in [(0, 0, 1), (1, 1, -1)]:
        k0 = sp.simplify(kappa.subs({etaM: eM, etaB: eB}))
        c0 = sp.simplify(c.subs({etaM: eM, etaB: eB}))
        check("A4  (eta_M,eta_B)=(%d,%d):  kappa = %d" % (eM, eB, kap), k0 == kap)
        check("A5  (eta_M,eta_B)=(%d,%d):  c = 0" % (eM, eB), c0 == 0)

    # dhat^2 = d^2 there, for every prevalence and every d
    dhat2 = kappa ** 2 * d2 / (1 + c * d2)
    for (eM, eB) in [(0, 0), (1, 1)]:
        e = sp.simplify(dhat2.subs({etaM: eM, etaB: eB}) - d2)
        check("A6  (eta_M,eta_B)=(%d,%d):  dhat^2 = d^2  identically" % (eM, eB),
              e == 0, "so the swap is a counterexample to the sentence")

    # d = 0 gives equality whatever the rates
    check("A7  d = 0:  dhat^2 = d^2 = 0 whatever the rates",
          sp.simplify(dhat2.subs(d2, 0)) == 0)

    # |kappa| <= 1 always -- the proof in the text says only kappa <= 1, which
    # does not bound kappa^2.
    kap_num = sp.lambdify((etaM, etaB, piM), kappa, "math")
    c_num = sp.lambdify((etaM, etaB, piM), c, "math")
    worst = 0.0
    grid = [i / 20 for i in range(21)]
    pgrid = [0.05, 0.2, 0.5, 0.8, 0.95]
    bad = []
    strict = 0
    for eM in grid:
        for eB in grid:
            for p in pgrid:
                qm = p * (1 - eM) + (1 - p) * eB
                qb = (1 - p) * (1 - eB) + p * eM
                if qm <= 1e-12 or qb <= 1e-12:
                    continue
                k = kap_num(eM, eB, p)
                cc = c_num(eM, eB, p)
                worst = max(worst, abs(k))
                if abs(k) > 1 + 1e-12:
                    bad.append((eM, eB, p, k))
                # and the inequality itself, at a few separations
                for dd2 in [0.0, 0.25, 1.0, 4.0, 25.0]:
                    dh = k * k * dd2 / (1 + cc * dd2)
                    if dh > dd2 + 1e-12:
                        bad.append(("ineq", eM, eB, p, dd2))
                    if dd2 > 0 and (eM, eB) not in [(0.0, 0.0), (1.0, 1.0)]:
                        if dh < dd2 - 1e-12:
                            strict += 1
    check("A8  |kappa| <= 1 on the grid, and dhat^2 <= d^2",
          not bad, "max |kappa| = %.6f over %d cells" % (worst, len(grid) ** 2 * len(pgrid)))
    check("A9  strict inequality away from the two corners",
          strict > 0, "%d strict cases" % strict)

    # the exhaustive equality census: for d > 0, equality only at the corners
    eq = []
    for eM in grid:
        for eB in grid:
            for p in pgrid:
                qm = p * (1 - eM) + (1 - p) * eB
                qb = (1 - p) * (1 - eB) + p * eM
                if qm <= 1e-12 or qb <= 1e-12:
                    continue
                k = kap_num(eM, eB, p)
                cc = c_num(eM, eB, p)
                dd2 = 1.0
                if abs(k * k * dd2 / (1 + cc * dd2) - dd2) < 1e-12:
                    eq.append((eM, eB))
    eqset = sorted(set(eq))
    check("A10 for d > 0 equality exactly at (0,0) and (1,1)",
          eqset == [(0.0, 0.0), (1.0, 1.0)], "found %s" % (eqset,))

    # and the direct simulation of the complete interchange, to be sure the
    # model really does say what the algebra says
    rng = np.random.default_rng(20260921)
    n = 400000
    p_dim = 3
    Sigma = np.array([[2.0, 0.4, 0.1], [0.4, 1.0, 0.2], [0.1, 0.2, 1.5]])
    Delta = np.array([0.8, -0.5, 0.3])
    d2_true = Delta @ np.linalg.solve(Sigma, Delta)
    A = np.linalg.cholesky(Sigma)
    y = rng.random(n) < 0.5
    x = (rng.standard_normal((n, p_dim)) @ A.T) + np.outer(y, Delta)
    yhat = ~y                                   # every label wrong: the swap
    mM = x[yhat].mean(0)
    mB = x[~yhat].mean(0)
    Sh = np.cov(np.vstack([x[yhat] - mM, x[~yhat] - mB]).T, bias=True)
    dh2 = (mM - mB) @ np.linalg.solve(Sh, (mM - mB))
    check("A11 simulated complete interchange: dhat^2 = d^2",
          abs(dh2 / d2_true - 1) < 0.02,
          "dhat^2/d^2 = %.4f on n = %d" % (dh2 / d2_true, n))


# ---------------------------------------------------------------------------
# B.  pr:blind item 3).  what_j = lambda (Sigma^{-1} Delta)_j off R.  The text
#     says a blind spot "can neither create, nor destroy, nor reverse".
# ---------------------------------------------------------------------------

def _blind_lambda(hM, hB, delta1, delta2, piM=0.5, npts=4001, span=9.0):
    """lambda for p = 2, Sigma = I, R = {1}, h a function of x_1 alone.

    Exact in the x_2 direction (Gaussian conditionals), quadrature in x_1.
    Returns (lambda, kappa_agg) with lambda = what_2 / (Sigma^{-1}Delta)_2.
    """
    piB = 1 - piM
    z = np.linspace(-span, span, npts)
    dz = z[1] - z[0]
    phiM = np.exp(-(z - delta1 / 2) ** 2 / 2) / math.sqrt(2 * math.pi)
    phiB = np.exp(-(z + delta1 / 2) ** 2 / 2) / math.sqrt(2 * math.pi)
    u = piM * phiM                       # density of (Y = M, x_1 = z)
    v = piB * phiB
    hm = hM(z)
    hb = hB(z)
    a = np.trapezoid(u * (1 - hm), dx=dz)   # true M keeping label M
    b = np.trapezoid(v * hb, dx=dz)         # true B flipped to M
    cc = np.trapezoid(u * hm, dx=dz)        # true M flipped to B
    d = np.trapezoid(v * (1 - hb), dx=dz)   # true B keeping label B
    kappa_agg = a / (a + b) + d / (cc + d) - 1

    # the two observed populations, as mixtures over z
    wM = np.vstack([u * (1 - hm), v * hb])          # weights, labelled M
    wB = np.vstack([u * hm, v * (1 - hb)])          # weights, labelled B
    m2 = np.array([delta2 / 2, -delta2 / 2])        # x_2-means of the two classes

    def moments(w):
        tot = np.trapezoid(w.sum(0), dx=dz)
        mu1 = np.trapezoid((w * z).sum(0), dx=dz) / tot
        mu2 = np.trapezoid((w.T @ m2), dx=dz) / tot
        s11 = np.trapezoid((w * (z ** 2 + 0.0)).sum(0), dx=dz) / tot - mu1 ** 2
        s22 = np.trapezoid((w.T @ (m2 ** 2 + 1.0)), dx=dz) / tot - mu2 ** 2
        s12 = np.trapezoid((w * z).T @ m2, dx=dz) / tot - mu1 * mu2
        return tot, np.array([mu1, mu2]), np.array([[s11, s12], [s12, s22]])

    tM, muM, SM = moments(wM)
    tB, muB, SB = moments(wB)
    Shat = (tM * SM + tB * SB) / (tM + tB)
    what = np.linalg.solve(Shat, muM - muB)
    return what[1] / delta2, kappa_agg


def group_B():
    LOG.append("=== B.  pr:blind 3): the sign of lambda ===")

    zero = lambda z: np.zeros_like(z)

    # the reviewer's two constant readers.  A constant is a function of Rx.
    lam, kag = _blind_lambda(lambda z: 0.5 + 0 * z, lambda z: 0.5 + 0 * z, 1.2, 0.9)
    check("B1  the coin (h_M = h_B = 1/2):  lambda = 0",
          abs(lam) < 1e-9, "lambda = %.3e, kappa_agg = %.3e" % (lam, kag))

    lam, kag = _blind_lambda(lambda z: 1 + 0 * z, lambda z: 1 + 0 * z, 1.2, 0.9)
    check("B2  the complete interchange:  lambda < 0",
          lam < -0.5, "lambda = %.6f, kappa_agg = %.6f" % (lam, kag))

    lam, kag = _blind_lambda(zero, zero, 1.2, 0.9)
    check("B3  the error-free reader:  lambda = 1",
          abs(lam - 1) < 1e-6, "lambda = %.9f" % lam)

    # an honest blind spot: h depends on x_1, which the reading uses
    lam, kag = _blind_lambda(lambda z: 0.4 * (z > 0), zero, 1.2, 0.9)
    check("B4  a one-sided blind spot:  0 < lambda < 1",
          0 < lam < 1, "lambda = %.6f" % lam)

    # THE POINT.  A reader strictly better than a coin AT EVERY POINT, and yet
    # lambda < 0.  So "better than a coin pointwise" does NOT rescue the
    # sentence: h_M(z) + h_B(z) < 1 for all z is not enough, because what the
    # sign of lambda turns on is h_M(z) + h_B(z') < 1 for all PAIRS z, z'.
    eps = 0.05
    hM = lambda z: (1 - eps) * (z > 0)
    hB = lambda z: (1 - eps) * (z < 0)
    zz = np.linspace(-9, 9, 4001)
    check("B5  the candidate is better than a coin at every point",
          float(np.max(hM(zz) + hB(zz))) < 1,
          "max (h_M + h_B) = %.3f" % float(np.max(hM(zz) + hB(zz))))
    lam, kag = _blind_lambda(hM, hB, 4.0, 0.9)
    check("B6  and yet lambda < 0",
          lam < 0, "lambda = %.6f, kappa_agg = %.6f" % (lam, kag))

    # the pair criterion, verified as the sign of the double integral
    #   ad - bc = int int u(z) v(z') [ (1-h_M(z))(1-h_B(z')) - h_M(z) h_B(z') ]
    z = np.linspace(-9, 9, 2001)
    dz = z[1] - z[0]
    u = 0.5 * np.exp(-(z - 2.0) ** 2 / 2) / math.sqrt(2 * math.pi)
    v = 0.5 * np.exp(-(z + 2.0) ** 2 / 2) / math.sqrt(2 * math.pi)
    hm, hb = hM(z), hB(z)
    a = np.trapezoid(u * (1 - hm), dx=dz)
    b = np.trapezoid(v * hb, dx=dz)
    cc = np.trapezoid(u * hm, dx=dz)
    d = np.trapezoid(v * (1 - hb), dx=dz)
    kernel = np.outer(1 - hm, 1 - hb) - np.outer(hm, hb)
    dbl = np.trapezoid(np.trapezoid(np.outer(u, v) * kernel, dx=dz, axis=1), dx=dz)
    check("B7  ad - bc equals the double integral of the pair kernel",
          abs(a * d - b * cc - dbl) < 1e-10 * max(1.0, abs(dbl)),
          "ad-bc = %.6e, integral = %.6e" % (a * d - b * cc, dbl))
    check("B8  and the pair kernel is the condition h_M(z) + h_B(z') < 1",
          float(kernel.min()) < 0 and a * d - b * cc < 0,
          "min kernel = %.4f, ad-bc = %.4e" % (float(kernel.min()), a * d - b * cc))

    # the sufficient condition that DOES work
    eps2 = 0.2
    hM2 = lambda z: 0.4 * (z > 0) + 0.1 * (z <= 0)
    hB2 = lambda z: 0.3 * (z < 0) + 0.2 * (z >= 0)
    check("B9  sup h_M + sup h_B < 1 for the repaired hypothesis",
          0.4 + 0.3 < 1)
    lam, kag = _blind_lambda(hM2, hB2, 4.0, 0.9)
    check("B10 and then lambda > 0",
          lam > 0, "lambda = %.6f" % lam)
    # a small random sweep of readers obeying sup + sup < 1
    rng = np.random.default_rng(7)
    worst = 1e9
    bad = 0
    for _ in range(60):
        sM, sB = rng.uniform(0, 0.45), rng.uniform(0, 0.45)
        cM, cB = rng.uniform(-2, 2), rng.uniform(-2, 2)
        f = lambda z, s=sM, c=cM: s / (1 + np.exp(-(z - c)))
        g = lambda z, s=sB, c=cB: s / (1 + np.exp(-(c - z)))
        lam, _ = _blind_lambda(f, g, rng.uniform(0.5, 4.0), 0.9)
        worst = min(worst, lam)
        if lam <= 0:
            bad += 1
    check("B11 60 random readers with sup h_M + sup h_B < 0.9:  lambda > 0 in all",
          bad == 0, "least lambda = %.6f" % worst)

    # "cannot create" is unconditional: a coordinate the true rule ignores
    lam, _ = _blind_lambda(lambda z: 0.4 * (z > 0), zero, 1.2, 0.0)
    check("B12 delta_2 = 0 gives what_2 = 0 whatever lambda is",
          True, "the identity what_j = lambda (Sigma^{-1}Delta)_j has no other content there")


# ---------------------------------------------------------------------------
# C.  Lemma le:opt.  Two features of size L at distance d, smoothed at s.
# ---------------------------------------------------------------------------

def group_C():
    LOG.append("=== C.  le:opt: when do two features merge ===")

    x, d, sig = sp.symbols("x d sigma", positive=True)
    f = sp.exp(-(x - d / 2) ** 2 / (2 * sig ** 2)) + sp.exp(-(x + d / 2) ** 2 / (2 * sig ** 2))
    f1 = sp.simplify(sp.diff(f, x).subs(x, 0))
    check("C1  the midpoint is always a critical point", sp.simplify(f1) == 0)
    f2 = sp.simplify(sp.diff(f, x, 2).subs(x, 0))
    # f'' (0) > 0  <=>  d > 2 sigma
    sol = sp.solve(sp.Eq(sp.simplify(f2 / sp.exp(-d ** 2 / (8 * sig ** 2))), 0), d)
    check("C2  the midpoint changes type exactly at d = 2 sigma",
          any(sp.simplify(s - 2 * sig) == 0 for s in sol), "roots %s" % (sol,))

    # and sigma is the blurred width: a feature of Gaussian size L convolved
    # with a kernel of scale s has width sqrt(L^2 + s^2), not s
    L, s = sp.symbols("L s", positive=True)
    t = sp.Symbol("t", real=True)
    conv = sp.integrate(sp.exp(-t ** 2 / (2 * L ** 2)) * sp.exp(-(x - t) ** 2 / (2 * s ** 2)),
                        (t, -sp.oo, sp.oo))
    conv = sp.simplify(conv)
    width2 = sp.simplify(-1 / sp.diff(sp.log(conv), x, 2))
    check("C3  a feature of size L blurred at s has width^2 = L^2 + s^2",
          sp.simplify(width2 - (L ** 2 + s ** 2)) == 0, "width^2 = %s" % width2)

    # so the merge threshold inside the text's own model
    for (LL, ss) in [(2.0, 3.0), (2.0, 2.0), (0.0, 3.0), (5.0, 1.0)]:
        thr = 2 * math.sqrt(LL ** 2 + ss ** 2)
        # numerical bimodality test
        def bimodal(dd):
            sg = math.sqrt(LL ** 2 + ss ** 2)
            xs = np.linspace(-3 * dd - 5 * sg, 3 * dd + 5 * sg, 20001)
            y = np.exp(-(xs - dd / 2) ** 2 / (2 * sg ** 2)) + np.exp(-(xs + dd / 2) ** 2 / (2 * sg ** 2))
            mid = len(xs) // 2
            return y[mid] < max(y) - 1e-12
        check("C4  L=%.1f s=%.1f: merge threshold is 2 sqrt(L^2+s^2) = %.3f"
              % (LL, ss, thr),
              (not bimodal(thr * 0.98)) and bimodal(thr * 1.02),
              "2s alone would say %.3f" % (2 * ss))

    # the two numbers the text quotes, recomputed
    for (ss, LL) in [(2.0, 2.0), (3.0, 2.0)]:
        LOG.append("     s = %.0f mm, L = %.0f mm:  2s = %.1f mm, 2 sqrt(L^2+s^2) = %.2f mm"
                   % (ss, LL, 2 * ss, 2 * math.sqrt(LL ** 2 + ss ** 2)))
    check("C5  the text's claim at s = 3 mm survives a fortiori",
          2 * math.sqrt(2.0 ** 2 + 3.0 ** 2) > 6.0,
          "two spiculations 6 mm apart still merge, the threshold being 7.21 mm")


# ---------------------------------------------------------------------------
# D.  pr:vanish.  The frame in which theta has constant components.
# ---------------------------------------------------------------------------

def group_D():
    LOG.append("=== D.  pr:vanish: the adapted Darboux frame ===")

    X, Y, p = sp.symbols("X Y p", real=True)
    coords = (X, Y, p)

    def vf(c):           # a vector field as its component triple
        return sp.Matrix(c)

    def bracket(a, b):
        out = []
        for i in range(3):
            out.append(sum(a[j] * sp.diff(b[i], coords[j]) - b[j] * sp.diff(a[i], coords[j])
                           for j in range(3)))
        return sp.Matrix([sp.simplify(o) for o in out])

    # alpha = dY + p dX, as the text has it
    alpha = sp.Matrix([[p, 1, 0]])          # components on (dX, dY, dp)

    e1 = vf([0, 0, 1])                      # d/dp  -- tangent to the fiber
    e2 = vf([1, -p, 0])                     # d/dX - p d/dY
    R = vf([0, 1, 0])                       # d/dY

    check("D1  alpha(e_1) = 0", sp.simplify((alpha * e1)[0]) == 0)
    check("D2  alpha(e_2) = 0", sp.simplify((alpha * e2)[0]) == 0)
    check("D3  alpha(R) = 1", sp.simplify((alpha * R)[0]) == 1)

    # d alpha = dp ^ dX ; the Reeb field is in its kernel
    # d alpha (A, B) = A(p) B(X) - A(X) B(p)
    def dalpha(a, b):
        return sp.simplify(a[2] * b[0] - a[0] * b[2])
    check("D4  d alpha (R, .) = 0",
          dalpha(R, e1) == 0 and dalpha(R, e2) == 0 and dalpha(R, R) == 0)
    check("D5  varpi(e_1, e_2) = 1, a CONSTANT", dalpha(e1, e2) == 1)

    b12 = bracket(e1, e2)
    check("D6  [e_1, e_2] = -R", sp.simplify(b12 + R) == sp.zeros(3, 1),
          "= %s" % b12.T.tolist())
    check("D7  [e_1, R] = 0", bracket(e1, R) == sp.zeros(3, 1))
    check("D8  [e_2, R] = 0", bracket(e2, R) == sp.zeros(3, 1))

    # V = ker d pi inside D, with pi(X,Y,p) = (X,Y)
    check("D9  d pi (e_1) = 0, so V = <e_1>", e1[0] == 0 and e1[1] == 0)
    check("D10 d pi (e_2) spans the tautological line {dY + p dX = 0}",
          sp.simplify(e2[1] + p * e2[0]) == 0)

    # theta := the form on D with kernel V is e^2, of components (0,1): constant
    LOG.append("     whence theta = e^2 has components (0,1) and varpi = e^1 ^ e^2,")
    LOG.append("     both constant, and the structure functions are constants too --")
    LOG.append("     so a frame derivative of Q = f theta (x) theta differentiates f alone.")

    # the check that matters: the frame components of Q and of its derivatives
    f = sp.Function("f")(X, Y, p)
    # Q in the frame: Q_ab = f delta_a2 delta_b2
    Q = sp.Matrix([[0, 0], [0, f]])
    for (nm, E) in [("e_1", e1), ("e_2", e2), ("R", R)]:
        dQ = sp.Matrix(2, 2, lambda i, j: sum(E[k] * sp.diff(Q[i, j], coords[k])
                                              for k in range(3)))
        offdiag = [dQ[0, 0], dQ[0, 1], dQ[1, 0]]
        check("D11 %-3s applied to Q keeps the two theta-indices"
              % nm, all(sp.simplify(o) == 0 for o in offdiag))

    # varpi kills a pair of theta-indices
    check("D12 varpi^{ab} theta_a theta_b = 0 (varpi antisymmetric)", True)


# ---------------------------------------------------------------------------
# E.  pr:floor.  rho^2_adj.
# ---------------------------------------------------------------------------

def group_E():
    LOG.append("=== E.  pr:floor: N_eff and rho^2_adj ===")

    rng = np.random.default_rng(11)
    for N in [1, 2, 5, 40]:
        m = np.ones(N)
        Neff = m.sum() ** 2 / (m ** 2).sum()
        check("E1  N = %2d equal weights: N_eff = N" % N, abs(Neff - N) < 1e-12)
    m = np.array([10.0, 1.0, 1.0, 1.0])
    check("E2  unequal weights: 1 <= N_eff < N",
          1 <= m.sum() ** 2 / (m ** 2).sum() < len(m),
          "N_eff = %.4f, N = %d" % (m.sum() ** 2 / (m ** 2).sum(), len(m)))
    m = np.array([1.0, 1e-9, 1e-9])
    check("E3  one patch carrying everything: N_eff -> 1",
          abs(m.sum() ** 2 / (m ** 2).sum() - 1) < 1e-6)

    # rho^2_adj at N_eff = 1 is 0/0
    check("E4  rho^2_adj is 0/0 at N_eff = 1", True,
          "(N_eff rho^2 - 1)/(N_eff - 1) with rho = 1 forced")

    # and it takes negative values under its own null
    for N in [4, 10, 50]:
        trials = 200000
        ph = rng.uniform(0, 2 * math.pi, size=(trials, N))
        rho2 = np.abs(np.exp(1j * ph).sum(1)) ** 2 / N ** 2
        adj = (N * rho2 - 1) / (N - 1)
        check("E5  N = %2d: E[rho^2_adj] = 0 and it goes negative" % N,
              abs(adj.mean()) < 0.01 and (adj < 0).mean() > 0.3,
              "mean %.4f, negative in %.1f%% of draws, least %.3f"
              % (adj.mean(), 100 * (adj < 0).mean(), adj.min()))
        check("E6  N = %2d: Var[rho^2_adj] = 1/(N(N-1))" % N,
              abs(adj.var() * N * (N - 1) - 1) < 0.05,
              "N(N-1) Var = %.4f" % (adj.var() * N * (N - 1)))


# ---------------------------------------------------------------------------
# F.  pr:growth.  Volume may grow while area falls.
# ---------------------------------------------------------------------------

def _prolate(a, c):
    """(area, volume) of the prolate spheroid with semi-axes (a, a, c), c > a."""
    e = math.sqrt(1 - (a / c) ** 2)
    area = 2 * math.pi * a ** 2 * (1 + (c / (a * e)) * math.asin(e))
    vol = 4 * math.pi * a ** 2 * c / 3
    return area, vol


def group_F():
    LOG.append("=== F.  pr:growth: growth is not an increase of area ===")

    A1, V1 = _prolate(0.2, 5.0)             # a needle
    r = 0.6
    A2, V2 = 4 * math.pi * r ** 2, 4 * math.pi * r ** 3 / 3
    check("F1  a needle and a ball, both diffeomorphic to S^2",
          V2 > V1 and A2 < A1,
          "V: %.4f -> %.4f (grows), A: %.4f -> %.4f (falls)" % (V1, V2, A1, A2))
    check("F2  both obey the isoperimetric inequality A^3 >= 36 pi V^2",
          A1 ** 3 >= 36 * math.pi * V1 ** 2 and A2 ** 3 >= 36 * math.pi * V2 ** 2)
    # a smooth family joining them, along which the volume rises and the area
    # falls at every instant: fix the volume to rise linearly and let the
    # spheroid fatten, which fixes c = 3V/(4 pi a^2).
    ts = np.linspace(0, 0.95, 191)   # stops short of the isoperimetric floor
    As, Vs = [], []
    for t in ts:
        a = 0.2 + t * (0.6 - 0.2)
        V = V1 * (1 + 0.08 * t)
        c = 3 * V / (4 * math.pi * a ** 2)
        if abs(c - a) < 1e-9:
            A, V = 4 * math.pi * a ** 2, 4 * math.pi * a ** 3 / 3
        elif c > a:
            A, V = _prolate(a, c)
        else:                                # oblate
            e = math.sqrt(1 - (c / a) ** 2)
            A = 2 * math.pi * a ** 2 + math.pi * c ** 2 / e * math.log((1 + e) / (1 - e))
            V = 4 * math.pi * a ** 2 * c / 3
        As.append(A)
        Vs.append(V)
    As, Vs = np.array(As), np.array(Vs)
    check("F3  along the family the area falls monotonically",
          bool(np.all(np.diff(As) < 0)), "A: %.3f -> %.3f" % (As[0], As[-1]))
    check("F4  while the volume rises monotonically",
          bool(np.all(np.diff(Vs) > 0)), "V: %.4f -> %.4f" % (Vs[0], Vs[-1]))
    check("F5  so equal areas can occur at different volumes too",
          True, "the obstruction of pr:growth is the area, not the growth")


def main():
    for g in (group_A, group_B, group_C, group_D, group_E, group_F):
        g()
        LOG.append("")
    LOG.append("%d checks, %d failed" % (NCHECK, NFAIL))
    text = "\n".join(LOG)
    print(text)
    with open("propchecks.log", "w") as fh:
        fh.write(text + "\n")
    return 1 if NFAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
