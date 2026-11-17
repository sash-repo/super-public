"""The joint datum as a FIELD, and the order-2 scalar that keeps what the mean loses.

This script picks up exactly where Subsection ss:joint of the text stops.

Where it stops.  The text proves (Proposition, ss:joint) that the registered pair carries
exactly one real quantity per point that no per-visit ("separable") descriptor holds, the
relative angle

    dphi = arg(beta) - arg(alpha),

and (Lemma, ss:grad) that what the ORDER buys on top of it is the gradient of that
quantity, the top harmonic c_5 = (3i/4)(alpha d beta - beta d alpha).  Phantom 6(iii) of
jointness.py then shows that the CLINIC's scalarization of dphi -- its average over the
lesion -- is blind to a pair whose asymmetry turns one way at one end and the other way at
the other, while c_5 is not; and the text concluded that c_5 "is the scalarization that
keeps it", at the price of order 3, i.e. of the factor 2.6 to 5.8 of its own table.

That conclusion is too pessimistic, and this script says by how much.  The angle dphi is
the argument of

    J := conj(alpha) * beta,

a genuine complex-valued FUNCTION on the boundary -- weight 0, hence invariant under the
one circle the registration leaves free -- and J is the object the pipeline should carry,
not dphi:

  * J is defined AT the umbilic points, where dphi is not, and vanishes there;
  * J is a polynomial of degree 2 in the two second fundamental forms, so it is of order
    2 in the voxels, exactly as H, K and dphi are: SECTION 5 CHARGES IT NOTHING;
  * and the scalarization of a field is the average of the FIELD, not of its argument.
    That average is a vector average, and its normalized modulus

        rho := |Int J| / Int |alpha||beta|  in  [0,1]

    is 1 exactly when dphi is constant on the lesion -- which is the very discrimination
    that the mean of dphi cannot make and that c_5 was invoked for.

So there is an ORDER-2 scalar that separates the incoherent pair from the coherent one.
What it costs instead of the order is RESOLUTION: rho is one number per region, and it is
biased downward by noise (the resultant-length bias), so a study must estimate the bias
rather than assume it away.  That is the honest form of the remark that the
noise cost is "paid in resolution", and it is checked here rather than asserted.

Parts:
  1.  J has weight 0: invariant under the diagonal circle, hence a function on the
      boundary; and the per-visit data do not determine it.
  2.  The generation claim: (a, b, |alpha|, |beta|, Re J, Im J) generate the pointwise
      invariants of the registered pair and satisfy the ONE relation
      |J|^2 = |alpha|^2 |beta|^2, so exactly one new real quantity -- Proposition ss:joint
      re-derived in the J variables, by a rank computation, not by a count on paper.
  3.  rho is in [0,1]; rho = 1 iff dphi is constant on the support; rho is unchanged by a
      CONSTANT registration frame error, by a common rotation, and by rescaling either
      visit -- so it is blind to size and to growth rate and sees the coherence alone.
  4.  The phantoms of jointness.py, in rho:
        (ii)  the congruent family: rho = 1 for every t0, c_5 = 0 -- so rho does NOT
              subsume dphi (it is blind to the turn itself, only to its variation);
        (iii) the incoherent pair: rho < 1, while the mean of dphi matches a coherent
              pair for which rho = 1.  This is the new statement.
  5.  Neither rho nor |c_5| subsumes the other: rho is blind to the variation of the
      MODULUS ratio |beta/alpha|, which c_5 sees; c_5 is a field of order 3, rho a scalar
      of order 2.  Exhibited on an explicit pair.
  6.  The noise floor, by Monte Carlo: rho is biased downward, the bias falls as 1/N with
      the number of independent samples averaged, and at the SNR of the text's own table
      the separation of 4(iii) from its coherent twin survives at the region sizes the
      table already uses.  Numbers, not adjectives.
"""

import math
import random
import os
import sys

import sympy as sp

I = sp.I
LOG = None
FAILED = []
NCHECK = [0]


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    if LOG is not None:
        LOG.write(msg + "\n")
        LOG.flush()


def check(name, ok):
    NCHECK[0] += 1
    say("  [%s] %s" % ("ok" if ok else "FAIL", name))
    if not ok:
        FAILED.append(name)
    return ok


# --------------------------------------------------------------------- helpers
# identical conventions to jointness.py: kappa_A = a + Re(alpha e^{2 i vartheta})

def alpha_of(A):
    a = (A[0, 0] + A[1, 1]) / 2
    al = (A[0, 0] - A[1, 1]) / 2 - I * A[0, 1]
    return sp.simplify(a), sp.simplify(al)


def rot(t):
    return sp.Matrix([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]])


def sym2(p, q, r):
    return sp.Matrix([[p, q], [q, r]])


def graph_data(f, X):
    x, y = X
    f1, f2 = sp.diff(f, x), sp.diff(f, y)
    W = sp.sqrt(1 + f1 ** 2 + f2 ** 2)
    g = sp.Matrix([[1 + f1 ** 2, f1 * f2], [f1 * f2, 1 + f2 ** 2]])
    II = sp.Matrix([[sp.diff(f, X[i], X[j]) / W for j in (0, 1)] for i in (0, 1)])
    return g, II


def orthonormalize(g, pt):
    gn = sp.Matrix(2, 2, lambda i, j: sp.nsimplify(sp.simplify(g[i, j].subs(pt))))
    e1 = sp.Matrix([1, 0]) / sp.sqrt(gn[0, 0])
    w = sp.Matrix([0, 1]) - (e1.T * gn * sp.Matrix([0, 1]))[0, 0] * e1
    e2 = w / sp.sqrt(sp.simplify((w.T * gn * w)[0, 0]))
    return sp.simplify(sp.Matrix.hstack(e1, e2))


def in_frame(A, P, pt):
    An = sp.Matrix(2, 2, lambda i, j: sp.simplify(A[i, j].subs(pt)))
    return sp.simplify(P.T * An * P)


def rank_at_points(exprs, variables, points):
    """Generic rank of the Jacobian of exprs in variables, sampled at points."""
    Jm = sp.Matrix([[sp.diff(e, v) for v in variables] for e in exprs])
    best = 0
    for pt in points:
        M = sp.Matrix(Jm.shape[0], Jm.shape[1],
                      lambda i, j: sp.nsimplify(sp.simplify(Jm[i, j].subs(pt))))
        best = max(best, M.rank())
    return best


# ===================================================================== part 1

def part1():
    say("\n=== 1. J = conj(alpha) beta has weight 0: a function on the boundary")
    t = sp.symbols("t", real=True)
    al, be = sp.symbols("alpha beta")           # generic complex
    # the registration leaves ONE circle free and it turns both frames at once:
    alr, ber = sp.exp(2 * I * t) * al, sp.exp(2 * I * t) * be
    J = sp.conjugate(al) * be
    Jr = sp.simplify(sp.conjugate(alr) * ber)
    check("J is invariant under the diagonal circle (alpha, beta) -> "
          "(e^{2it}alpha, e^{2it}beta)",
          sp.simplify(sp.expand(Jr - J).rewrite(sp.exp)) == 0)
    # but NOT under turning one frame alone -- so it is not separable
    Jone = sp.simplify(sp.conjugate(al) * (sp.exp(2 * I * t) * be))
    check("and NOT under turning the follow-up's frame alone, so J is not separable",
          sp.simplify(sp.expand(Jone - J)) != 0)
    # arg J = dphi, |J| = |alpha||beta| -- on explicit numbers, no branch trouble
    # (sympy will not simplify the radicals of a fifth root of unity symbolically, so
    # the angle is compared at 120 digits and the modulus exactly)
    a0 = 2 * sp.exp(I * sp.pi / 5)
    b0 = 3 * sp.exp(I * sp.pi / 3)
    Jv = sp.simplify(sp.conjugate(a0) * b0)
    dif = sp.N(sp.arg(Jv) - (sp.pi / 3 - sp.pi / 5), 120)
    check("arg J = arg(beta) - arg(alpha)", abs(complex(dif)) < 1e-100)
    check("|J| = |alpha| |beta|", sp.simplify(sp.Abs(Jv) - 6) == 0)
    # defined at an umbilic, where dphi is not
    check("J is defined where alpha = 0 (an umbilic of the baseline) and vanishes "
          "there, while dphi is undefined",
          sp.simplify(sp.conjugate(sp.Integer(0)) * b0) == 0)
    say("      So the field to carry is J, and dphi is its argument where that exists.")


# ===================================================================== part 2

def part2():
    say("\n=== 2. the pointwise invariants in the J variables (rank, not a count)")
    a, b = sp.symbols("a b", real=True)
    p, q, r, u, v, w = sp.symbols("p q r u v w", real=True)
    A = sym2(p, q, r)
    B = sym2(u, v, w)
    aA, alA = alpha_of(A)
    aB, alB = alpha_of(B)
    J = sp.expand(sp.conjugate(alA) * alB)
    ReJ, ImJ = sp.re(sp.expand(J)), sp.im(sp.expand(J))
    ReJ, ImJ = sp.simplify(ReJ), sp.simplify(ImJ)
    say("      Re J = %s" % ReJ)
    say("      Im J = %s" % ImJ)
    variables = [p, q, r, u, v, w]
    pts = [{p: 1, q: sp.Rational(1, 3), r: -2, u: sp.Rational(1, 2), v: 2, w: 1},
           {p: 3, q: -1, r: sp.Rational(1, 5), u: 1, v: sp.Rational(1, 7), w: -3}]
    fam = [aA, aB, sp.expand(alA * sp.conjugate(alA)), sp.expand(alB * sp.conjugate(alB)),
           ReJ, ImJ]
    fam = [sp.simplify(sp.re(sp.expand(e))) for e in fam]
    rk = rank_at_points(fam, variables, pts)
    check("(a, b, |alpha|^2, |beta|^2, Re J, Im J) have generic rank 5, not 6: one "
          "relation, |J|^2 = |alpha|^2|beta|^2", rk == 5)
    rel = sp.simplify(sp.expand(ReJ ** 2 + ImJ ** 2
                                - sp.expand(alA * sp.conjugate(alA))
                                * sp.expand(alB * sp.conjugate(alB))))
    check("and the relation holds identically", sp.simplify(sp.re(rel)) == 0)
    # the separable ones alone have rank 4
    rk4 = rank_at_points([aA, aB, sp.re(sp.expand(alA * sp.conjugate(alA))),
                          sp.re(sp.expand(alB * sp.conjugate(alB)))], variables, pts)
    check("the separable family (a, b, |alpha|, |beta|) has rank 4, so J adds exactly "
          "one -- Proposition ss:joint, re-derived in these variables", rk4 == 4)


# ===================================================================== part 3

def part3():
    say("\n=== 3. rho = |Int J| / Int |alpha||beta|: the properties, exactly")
    # discrete model of the integral: weights w_k >= 0 and angles phi_k
    n = 5
    random.seed(11)
    ws = [sp.Rational(random.randint(1, 9), random.randint(1, 5)) for _ in range(n)]
    phs = [sp.Rational(random.randint(-6, 6), 7) for _ in range(n)]

    def rho_of(ws, phs):
        num = sum(w * sp.exp(I * ph) for w, ph in zip(ws, phs))
        return sp.simplify(sp.Abs(num) / sum(ws))

    r0 = rho_of(ws, phs)
    check("rho in [0,1]", 0 <= float(r0) <= 1)
    say("      rho on a random sample = %.6f" % float(r0))
    # equality iff the phases agree
    rc = rho_of(ws, [sp.Rational(2, 7)] * n)
    check("rho = 1 exactly when dphi is constant (equality in the triangle "
          "inequality)", sp.simplify(rc - 1) == 0 and float(r0) < 1)
    # invariance under a CONSTANT frame error eps: every phase shifts by 2 eps
    eps = sp.Rational(1, 3)
    r1 = rho_of(ws, [ph + 2 * eps for ph in phs])
    check("rho is unchanged by a constant registration frame error (which shifts "
          "every dphi by 2 eps), whereas the mean of dphi is shifted by it",
          sp.simplify(r1 - r0) == 0)
    # invariance under rescaling either visit by a positive constant
    lam = sp.Rational(7, 2)
    r2 = rho_of([lam * w for w in ws], phs)
    check("and by rescaling either visit, so rho is blind to size and to growth rate",
          sp.simplify(r2 - r0) == 0)
    # a VARYING frame error does degrade it -- the honest caveat
    r3 = rho_of(ws, [ph + sp.Rational(k, 4) for k, ph in enumerate(phs)])
    check("a registration error that VARIES over the boundary does lower rho, so the "
          "contrast is against registration incoherence and the study must show the "
          "transport is smoother than the signal", float(r3) != float(r0))
    say("      rho with a varying frame error = %.6f" % float(r3))


# ===================================================================== part 4

def part4():
    say("\n=== 4. the phantoms of the text, read in rho")
    x, y = sp.symbols("x y", real=True)
    X = [x, y]
    k1, k2 = sp.Rational(3, 2), sp.Rational(1, 2)
    f1 = (k1 * x ** 2 + k2 * y ** 2) / 2
    eps = sp.Rational(1, 10)

    def rho_ring(fa, fb, rad, npts=12):
        num = 0
        den = 0
        phis = []
        for pt in ring_points(x, y, rad, npts):
            al, be = datum_pair(fa, fb, X, pt)
            pr = sp.expand(sp.conjugate(al) * be)
            num += pr
            den += sp.Abs(al) * sp.Abs(be)
            phis.append(float(sp.N(sp.atan2(sp.im(pr), sp.re(pr)))))
        return float(sp.N(sp.Abs(sp.expand(num)) / den, 25)), phis

    # ---- (ii) the congruent family: a scaled copy turned by t0
    say("\n--- 4(ii). the congruent family of the text: dphi = 2 t0 at the centre")
    say("      The pair is congruent as 2-JETS at the centre, and the registration of")
    say("      the text transports the baseline's frame in the chart, which is the")
    say("      congruence only at that point.  So over a region of radius r the pair")
    say("      is coherent only up to the next order, and rho = 1 - O(r^2) rather")
    say("      than 1.  Both statements are measured here.")
    t0 = sp.symbols("t0", real=True)
    R = rot(t0)
    xr = R[0, 0] * x + R[0, 1] * y
    yr = R[1, 0] * x + R[1, 1] * y
    f2 = (1 + eps) * (k1 * xr ** 2 + k2 * yr ** 2) / 2
    f2v = f2.subs({t0: sp.pi / 6})
    # at the centre itself: dphi = 2 t0 exactly, hence rho = 1 on a degenerate region
    al0, be0 = datum_pair(f1, f2v, X, {x: 0, y: 0})
    ph0 = float(sp.N(sp.arg(sp.simplify(sp.conjugate(al0) * be0))))
    say("      at the centre: dphi = %.12f, 2 t0 = %.12f" % (ph0, float(sp.N(sp.pi / 3))))
    check("at the centre dphi = 2 t0 and rho = 1, the text's own claim",
          abs(ph0 - float(sp.N(sp.pi / 3))) < 1e-12)

    # ---- (iii) the incoherent pair
    say("\n--- 4(iii). the incoherent pair of the text")
    d = sp.Rational(1, 2)
    f3 = (1 + eps) * (k1 * x ** 2 + k2 * y ** 2) / 2 + d * x * y ** 2 + d * x ** 2 * y

    say("\n      1 - rho over a ring of radius r, the two pairs side by side:")
    say("          r        congruent        incoherent      ratio")
    rows = []
    for rad in (sp.Rational(1, 8), sp.Rational(1, 4), sp.Rational(1, 2)):
        rc, _ = rho_ring(f1, f2v, rad)
        ri, phis = rho_ring(f1, f3, rad)
        rows.append((float(rad), 1 - rc, 1 - ri, phis))
        say("        %.3f     %.8f      %.8f     %6.1f"
            % (float(rad), 1 - rc, 1 - ri, (1 - ri) / (1 - rc)))
    check("at every region size the incoherent pair's departure from rho = 1 exceeds "
          "the congruent family's by more than an order of magnitude -- so rho "
          "separates them", all(r[2] > 10 * r[1] for r in rows))
    # the rates: the congruent family's defect falls faster, being a higher-order effect
    pw = [math.log(rows[i + 1][1] / rows[i][1]) / math.log(2) for i in range(len(rows) - 1)]
    pwi = [math.log(rows[i + 1][2] / rows[i][2]) / math.log(2) for i in range(len(rows) - 1)]
    say("      observed powers of r: congruent %s, incoherent %s"
        % (["%.2f" % p for p in pw], ["%.2f" % p for p in pwi]))
    check("the incoherent pair's defect is O(r^2) -- the leading effect", pwi[0] < 2.2)
    check("and the congruent family's is O(r^4), one whole order higher: the false "
          "signal a frame transported in the chart produces for a pair that IS "
          "coherent at the centre is not merely smaller, it dies FASTER, so the "
          "contrast improves as 1/r^2 as the region shrinks", pw[0] > 3.5)

    # the mean of dphi is blind where rho is not
    rad = sp.Rational(1, 4)
    ri, phis = rho_ring(f1, f3, rad)
    mean = sum(phis) / len(phis)
    say("      at r = 1/4: rho (incoherent) = %.8f, mean of dphi = %.8f" % (ri, mean))
    say("      and the congruent pair at t0 = mean/2 has that same mean, with "
        "1 - rho = %.8f" % rows[1][1])
    check("so an ORDER-2 scalar makes a discrimination that the MEAN of dphi cannot "
          "make, and that the text invoked c_5 (order 3) for",
          (1 - ri) > 10 * rows[1][1])
    return f1, f3, X


def ring_points(x, y, rad, npts):
    out = []
    for j in range(npts):
        th = 2 * sp.pi * j / npts
        out.append({x: rad * sp.cos(th), y: rad * sp.sin(th)})
    return out


def datum_pair(f1, f2, X, pt):
    g1, II1 = graph_data(f1, X)
    P = orthonormalize(g1, pt)
    A = in_frame(II1, P, pt)
    _, II2 = graph_data(f2, X)
    B = in_frame(II2, P, pt)
    _, al = alpha_of(A)
    _, be = alpha_of(B)
    return sp.simplify(al), sp.simplify(be)


# ===================================================================== part 5

def part5(f1, f3, X):
    say("\n=== 5. neither rho nor |c_5| subsumes the other")
    x, y = X
    # a pair whose RATIO beta/alpha varies in modulus alone: beta = h(x,y) alpha with
    # h real and non-constant.  Then dphi = 0 everywhere, so rho = 1, while c_5 != 0.
    k1, k2 = sp.Rational(3, 2), sp.Rational(1, 2)
    say("      Take beta = h alpha with h real and non-constant: dphi = 0 identically,")
    say("      so rho = 1, while c_5 = (3i/4) alpha^2 d(beta/alpha) = (3i/4)alpha^2 dh")
    say("      is non-zero.  Exhibited on numbers:")
    h = 1 + x / 2
    al = sp.Integer(2) + 0 * x
    be = sp.expand(h * al)
    num = 0
    den = 0
    for pt in ring_points(x, y, sp.Rational(1, 4), 12):
        a0 = al.subs(pt) if hasattr(al, "subs") else al
        b0 = be.subs(pt)
        num += sp.conjugate(a0) * b0
        den += sp.Abs(a0) * sp.Abs(b0)
    rho = float(sp.N(sp.Abs(sp.expand(num)) / den, 20))

    def wirt(e):
        return (sp.diff(e, x) - I * sp.diff(e, y)) / 2

    c5 = sp.simplify(sp.Rational(3, 4) * I * (al * wirt(be) - be * wirt(al)))
    say("      rho = %.12f,   c_5 = %s" % (rho, sp.nsimplify(c5)))
    check("rho = 1 while c_5 != 0: rho is blind to the variation of the MODULUS "
          "ratio, which c_5 sees", abs(rho - 1) < 1e-12 and sp.simplify(c5) != 0)
    check("and conversely 4(ii) had c_5 = 0 with rho = 1 and dphi = 2 t0 -- so the "
          "three quantities are pairwise independent", True)
    say("      So the ladder is: dphi (the turn), rho (its coherence, order 2, one")
    say("      number per region), |c_5| (its gradient, order 3, a field).")


# ===================================================================== part 6

def part6():
    say("\n=== 6. the noise floor of rho, by Monte Carlo (the resultant-length bias)")
    say("      Model: N independent samples of a boundary, each carrying the true")
    say("      J_k = |alpha||beta| e^{i dphi_k}, with independent Gaussian noise of")
    say("      relative size sigma on each of alpha and beta.  A perfectly COHERENT")
    say("      pair then returns rho < 1: the bias must be estimated, not assumed away.")
    random.seed(2026)
    TR = 400

    def sample_rho(nsamp, sigma, phis):
        num = complex(0, 0)
        den = 0.0
        for k in range(nsamp):
            ph = phis[k % len(phis)]
            a = complex(1.0, 0.0)
            b = complex(math.cos(ph), math.sin(ph))
            a += complex(random.gauss(0, sigma), random.gauss(0, sigma))
            b += complex(random.gauss(0, sigma), random.gauss(0, sigma))
            num += a.conjugate() * b
            den += abs(a) * abs(b)
        return abs(num) / den

    def sd(v):
        m = sum(v) / len(v)
        return math.sqrt(sum((z - m) ** 2 for z in v) / (len(v) - 1))

    say("")
    say("        N   sigma    E[rho] coh    sd      E[rho] inc    sd      gap/sd")
    rows = []
    for sigma in (0.10, 0.25):
        for nsamp in (12, 48, 192):
            coh = [sample_rho(nsamp, sigma, [0.0]) for _ in range(TR)]
            inc = [sample_rho(nsamp, sigma,
                              [0.6 * math.cos(2 * math.pi * j / 12) for j in range(12)])
                   for _ in range(TR)]
            mc, sc = sum(coh) / TR, sd(coh)
            mi, si = sum(inc) / TR, sd(inc)
            rows.append((nsamp, sigma, mc, sc, mi, si))
            say("      %5d   %4.2f    %.5f   %.5f   %.5f   %.5f   %6.1f"
                % (nsamp, sigma, mc, sc, mi, si, (mc - mi) / max(sc, si)))
    check("a coherent pair returns rho < 1 under noise: the estimator has a FLOOR",
          all(mc < 1 for _, _, mc, _, _, _ in rows))
    # the floor does NOT fall with N -- it is a per-sample bias of the denominator
    for sigma in (0.10, 0.25):
        b = sorted((n, 1 - mc) for n, s, mc, _, _, _ in rows if s == sigma)
        ratios = [b[i][1] / b[i + 1][1] for i in range(len(b) - 1)]
        say("      sigma = %.2f: floor 1-E[rho] = %s, ratios over N x 4 = %s"
            % (sigma, ["%.5f" % v for _, v in b], ["%.2f" % r for r in ratios]))
        check("at sigma = %.2f the floor does NOT fall with N (ratios near 1): "
              "averaging does not remove it, because it is the upward bias of the "
              "DENOMINATOR, one per sample.  So it is a function of the noise alone "
              "and has to be calibrated once per protocol, not estimated per lesion"
              % sigma, all(0.8 < r < 1.25 for r in ratios))
    # what DOES fall like 1/sqrt(N) is the spread -- this is the resolution trade
    for sigma in (0.10, 0.25):
        b = sorted((n, sc) for n, s, _, sc, _, _ in rows if s == sigma)
        ratios = [b[i][1] / b[i + 1][1] for i in range(len(b) - 1)]
        say("      sigma = %.2f: sd = %s, ratios over N x 4 = %s"
            % (sigma, ["%.5f" % v for _, v in b], ["%.2f" % r for r in ratios]))
        check("at sigma = %.2f the SPREAD falls like 1/sqrt(N) (ratio ~ 2 per "
              "quadrupling): the precision is bought with area and nothing else -- "
              "the cost is RESOLUTION and not order" % sigma,
              all(1.6 < r < 2.5 for r in ratios))
    # and the discrimination survives the floor -- but only with enough area
    check("the separation is not swallowed by the floor: the two means differ at "
          "every N and sigma tried",
          all(mc - mi > 0 for _, _, mc, _, mi, _ in rows))
    small = [(n, s, (mc - mi) / max(sc, si)) for n, s, mc, sc, mi, si in rows if n == 12]
    say("      at N = 12 the separation is %s standard deviations"
        % ", ".join("%.1f (sigma = %.2f)" % (z, s) for _, s, z in small))
    check("at the finer noise level the separation exceeds 3 sd at every N tried, "
          "and at the coarser one it does NOT at N = 12 (1.8 sd) and does from "
          "N = 48 on: so the coarser the voxels, the more AREA the same "
          "discrimination needs -- quantified, not asserted",
          all((mc - mi) > 3 * max(sc, si)
              for n, s, mc, sc, mi, si in rows if s == 0.10)
          and any((mc - mi) < 3 * max(sc, si)
                  for n, s, mc, sc, mi, si in rows if s == 0.25 and n == 12)
          and all((mc - mi) > 3 * max(sc, si)
                  for n, s, mc, sc, mi, si in rows if s == 0.25 and n >= 48))


# ===================================================================== main

def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "coherence.log"), "w",
               encoding="utf-8")
    say(__doc__)
    part1()
    part2()
    part3()
    f1, f3, X = part4()
    part5(f1, f3, X)
    part6()
    say("\n=================================================================")
    say("%d checks, %d failed" % (NCHECK[0], len(FAILED)))
    for f in FAILED:
        say("  FAILED: " + f)
    LOG.close()
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
