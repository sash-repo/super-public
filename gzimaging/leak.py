"""Where the leak of Statement 5.5 comes from, and how it is undone.

H. Andersen gave a decomposition of the pulled-back lift under a CONSTANT stretch d phi = R U,
U = diag(l_1,l_2):

    tilde-phi^* Lambda_{g_2}(B)
        = Lambda_{g_1}(phi^* B)
          + lam * kappa_{phi^* B} * [ (l_1 l_2)^2 / |Uv|^6 - 1 ] * theta_V (x) theta_V ,

    |Uv|^2 = l_1^2 cos^2 th + l_2^2 sin^2 th ,

with the three-line proof: the horizontal lift is carried to |Uv| times the horizontal
lift of the image direction, so the horizontal component pulls back with NO multiplier;
the fiber map has d th'/d th = l_1 l_2 / |Uv|^2, which squares into the vertical
component; and both images stay pure, so nothing mixed appears.

Everything below is redone in OUR chart and from OUR lift, and as an identity of
TENSORS -- all three frame components separately -- which is stronger than checking the
pairing at sample points.  The consequences she draws are checked too:

  * the first term is the pairing of Statement 3.13 applied to the STRAINED pair
    (II_1, phi^* II_2), which is the four-forms datum of Prop. 5.3 item 5);
  * that term has harmonics {1,3,5} and nothing else, and the correction carries the
    whole leak, harmonics >= 7, and contaminates 1, 3, 5 as well;
  * "undone" acts on the ARGUMENT and not on the output: the inverse Blaschke factor
    applied to the answer recovers nothing;
  * for a stretch that VARIES, a mixed component appears, proportional to the
    derivative of d phi.

The chart is that of `lift.py` and `framefree.py`: Sigma = R^2 flat,
PT^*Sigma = {(x,y,m)}, alpha = dy - m dx, e_1 = d_m, e_2 = d_x + m d_y, R = d_y, and
the direction is v = (1,m)/sqrt(1+m^2), so m = tan theta.

Nothing is asserted below that the run does not check.
"""

import os
import numpy as np
import sympy as sp

LOG = None
CHECKS = [0, 0]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def check(name, ok):
    CHECKS[0] += 1
    CHECKS[1] += bool(ok)
    say(("  PASS  " if ok else "  FAIL  ") + name)
    return bool(ok)


x, y, m = sp.symbols("x y m")
lam = sp.Symbol("lam")                      # the free vertical constant of (11)
l1, l2 = sp.symbols("l1 l2", positive=True)


def e1(F):
    return sp.diff(F, m)


def e2(F):
    return sp.diff(F, x) + m * sp.diff(F, y)


def Rf(F):
    return sp.diff(F, y)


def kGz(F, G):
    """Equation (29) of Appendix E.2, as printed."""
    F2, F1, F0 = F
    G2, G1, G0 = G
    return sp.expand(
        + 2 * F1 * e2(e2(G2)) - 2 * F2 * e2(e2(G1)) - 4 * F0 * e1(e2(G2))
        + 4 * F2 * e1(e2(G0)) + 2 * F0 * e1(e1(G1)) - 2 * F1 * e1(e1(G0))
        + 4 * e2(F1) * e2(G2) - 4 * e2(F2) * e2(G1) - 2 * e2(F0) * e1(G2)
        - e2(F1) * e1(G1) + 6 * e2(F2) * e1(G0) + 2 * e2(e2(F1)) * G2
        - 2 * e2(e2(F2)) * G1 - 6 * e1(F0) * e2(G2) + e1(F1) * e2(G1)
        + 2 * e1(F2) * e2(G0) + 4 * e1(F0) * e1(G1) - 4 * e1(F1) * e1(G0)
        - 4 * e1(e2(F0)) * G2 + 4 * e1(e2(F2)) * G0 + 2 * e1(e1(F0)) * G1
        - 2 * e1(e1(F1)) * G0 + F1 * Rf(G1) - 4 * F2 * Rf(G0)
        + 4 * Rf(F0) * G2 - Rf(F1) * G1
    )


def jet(rng, order=2):
    """A random symmetric 2-tensor field on Sigma, a polynomial 2-jet."""
    out = []
    for _ in range(3):
        p = sp.S(0)
        for i in range(order + 1):
            for j in range(order + 1 - i):
                p += sp.Integer(int(rng.integers(-3, 4))) * x ** i * y ** j
        out.append(p)
    return out


def kappa(A, mv):
    """A(v,v) for the unit direction v = (1,m)/sqrt(1+m^2)."""
    A11, A12, A22 = A
    return (A11 + 2 * mv * A12 + mv ** 2 * A22) / (1 + mv ** 2)


def lift(A, mv, vert=None):
    """Lambda_g(A) of (11) in the frame (e_1,e_2): (F_2, F_1, F_0).

    theta_V = d theta and e_1 = d_m give d theta(e_1) = 1/(1+m^2), whence the
    (1+m^2)^{-2}; theta_H is dual to the horizontal lift of the UNIT direction and
    e_2 has length sqrt(1+m^2), whence the (1+m^2).
    """
    if vert is None:
        vert = lam
    kap = kappa(A, mv)
    return (vert * kap / (1 + mv ** 2) ** 2, sp.S(0), kap * (1 + mv ** 2))


def pullback(G, a, b, c):
    """Pull back a symmetric form with components G = (G_2,G_1,G_0) under

        e_1 |-> a e_1' ,     e_2 |-> b e_2' + c e_1' .

    P = [[a, c], [0, b]] in the bases (e_1',e_2'), and the form goes to P^T G' P.
    """
    G2, G1, G0 = G
    F2 = a ** 2 * G2
    F1 = a * (c * G2 + b * G1)
    F0 = c ** 2 * G2 + 2 * b * c * G1 + b ** 2 * G0
    return (sp.expand(F2), sp.expand(F1), sp.expand(F0))


def dtilde(phi1, phi2):
    """(a, b, c, M) of the lifted diffeomorphism tilde-phi of phi = (phi1,phi2).

    The direction (1,m) goes to (e_2 phi1, e_2 phi2), of slope M; and
    d tilde-phi(e_1) = (d_m M) e_1', d tilde-phi(e_2) = (e_2 phi1) e_2' + (e_2 M) e_1'.
    """
    M = sp.together(e2(phi2) / e2(phi1))
    return sp.cancel(e1(M)), sp.cancel(e2(phi1)), sp.cancel(e2(M)), sp.cancel(M)


# --------------------------------------------------------------------------------- 1

def part1():
    say("\n=== 1. the differential of the lifted diffeomorphism, in the frame")
    r = l2 / l1
    a, b, c, M = dtilde(l1 * x, l2 * y)
    check("constant stretch: the fiber map is m |-> r m, r = l_2/l_1",
          sp.simplify(M - r * m) == 0)
    check("constant stretch: d tilde-phi(e_1) = r e_1'", sp.simplify(a - r) == 0)
    check("constant stretch: d tilde-phi(e_2) = l_1 e_2'", sp.simplify(b - l1) == 0)
    check("constant stretch: NO mixed part, c = e_2(M) = 0", sp.simplify(c) == 0)
    # a non-affine one, for contrast
    a2, b2, c2, _ = dtilde(l1 * x + x ** 2 / 2, l2 * y)
    check("a stretch that varies over the base has c = e_2(M) =/= 0",
          sp.simplify(c2) != 0)
    say("  for phi = (l_1 x + x^2/2, l_2 y) the mixed coefficient is c = %s"
        % sp.simplify(c2))


# --------------------------------------------------------------------------------- 2

def part2():
    say("\n=== 2. the decomposition, as an identity of TENSORS")
    r = l2 / l1
    rng = np.random.default_rng(20260921)
    B = jet(rng)
    # tilde-phi^* Lambda_{g_2}(B): the lift at the image point, pulled back
    xp, yp, mp = l1 * x, l2 * y, r * m
    B_at = [q.subs({x: xp, y: yp}, simultaneous=True) for q in B]
    G_img = lift(B_at, mp)
    a, b, c, _ = dtilde(l1 * x, l2 * y)
    LHS = pullback(G_img, a, b, c)
    # Lambda_{g_1}(phi^* B): phi^* B = U^T B(phi(x)) U, U = diag(l_1,l_2)
    B11, B12, B22 = B_at
    PB = (l1 ** 2 * B11, l1 * l2 * B12, l2 ** 2 * B22)
    MAIN = lift(PB, m)
    # the correction, purely vertical
    Uv2 = (l1 ** 2 + l2 ** 2 * m ** 2) / (1 + m ** 2)
    corr_coeff = lam * kappa(PB, m) * ((l1 * l2) ** 2 / Uv2 ** 3 - 1)
    CORR = (corr_coeff / (1 + m ** 2) ** 2, sp.S(0), sp.S(0))
    for i, nm in ((0, "vertical"), (1, "mixed"), (2, "horizontal")):
        d = sp.simplify(sp.together(LHS[i] - MAIN[i] - CORR[i]))
        check("the %s component of tilde-phi^* Lambda_{g_2}(B) is that of "
              "Lambda_{g_1}(phi^* B) plus the correction" % nm, d == 0)
    check("|Uv|^2 = l_1^2 cos^2 th + l_2^2 sin^2 th in the chart",
          sp.simplify(Uv2 - (l1 ** 2 + l2 ** 2 * m ** 2) / (1 + m ** 2)) == 0)
    check("the correction is PURELY vertical (no mixed, no horizontal part)",
          CORR[1] == 0 and CORR[2] == 0)
    # and NOT at l_1 = l_2: at |Uv|^2 = ell^2 the bracket is ell^4/ell^6 - 1
    ell = sp.Symbol("ell", positive=True)
    iso = sp.simplify(sp.together(corr_coeff.subs({l1: ell, l2: ell})))
    check("the correction vanishes identically only for U = One, i.e. for an "
          "isometry -- NOT for every r = 1",
          sp.simplify(iso.subs(ell, 1)) == 0 and sp.simplify(iso) != 0)
    br = sp.simplify(((l1 * l2) ** 2 / Uv2 ** 3 - 1).subs({l1: ell, l2: ell}))
    check("under an isotropic dilation ell the bracket is the constant 1/ell^2 - 1, "
          "independent of the direction", sp.simplify(br - (1 / ell ** 2 - 1)) == 0
          and sp.simplify(sp.diff(br, m)) == 0)
    say("  see `iso.py`: that constant leaks into the value of the pairing and "
        "not into its spectrum.")
    say("  the correction coefficient is  lam * kappa_{phi^*B} * "
        "[ (l_1 l_2)^2 / |Uv|^6 - 1 ]")
    return LHS, MAIN, CORR


# --------------------------------------------------------------------------------- 3

def part3():
    say("\n=== 3. the three lines of the proof, each on its own")
    r = l2 / l1
    rng = np.random.default_rng(7)
    B = jet(rng)
    xp, yp, mp = l1 * x, l2 * y, r * m
    B_at = [q.subs({x: xp, y: yp}, simultaneous=True) for q in B]
    B11, B12, B22 = B_at
    PB = (l1 ** 2 * B11, l1 * l2 * B12, l2 ** 2 * B22)
    Uv2 = (l1 ** 2 + l2 ** 2 * m ** 2) / (1 + m ** 2)
    # (i) the horizontal lift is carried to |Uv| times the horizontal lift of the image
    #     direction: b = e_2(phi_1) is the length ratio of the UNIT frames
    a, b, c, _ = dtilde(l1 * x, l2 * y)
    ratio = b * sp.sqrt(1 + mp ** 2) / sp.sqrt(1 + m ** 2)
    check("d tilde-phi carries the horizontal lift of v to |Uv| times that of v'",
          sp.simplify(sp.powsimp(ratio ** 2 - Uv2, force=True)) == 0)
    # (ii) the horizontal component acquires NO multiplier
    kap_img = kappa(B_at, mp)
    check("hence kappa_B(v') * |Uv|^2 = II_2(Uv,Uv) = kappa_{phi^*B}(v): the "
          "horizontal component pulls back with no multiplier at all",
          sp.simplify(sp.together(kap_img * Uv2 - kappa(PB, m))) == 0)
    # (iii) the fiber map, and its derivative
    th, thp = sp.symbols("th thp", real=True)
    dthp = sp.simplify(sp.diff(sp.atan(r * sp.tan(th)), th))
    Uv2_th = l1 ** 2 * sp.cos(th) ** 2 + l2 ** 2 * sp.sin(th) ** 2
    check("the fiber map is tan th' = r tan th, with d th'/d th = l_1 l_2 / |Uv|^2",
          sp.simplify(dthp - l1 * l2 / Uv2_th) == 0)
    check("so theta_V pulls back by that factor, and theta_V (x) theta_V by its "
          "square, (l_1 l_2)^2 / |Uv|^4",
          sp.simplify((l1 * l2 / Uv2_th) ** 2
                      - (l1 * l2) ** 2 / Uv2_th ** 2) == 0)
    # (iv) the vertical coefficient before the jacobian is kappa_{phi^*B}/|Uv|^2
    check("the vertical coefficient of Lambda_{g_2}(B) at the image is "
          "kappa_{phi^*B} / |Uv|^2, whence the sixth power in the correction",
          sp.simplify(sp.together(kap_img - kappa(PB, m) / Uv2)) == 0)


# --------------------------------------------------------------------------------- 4

def spectrum(expr, pt, N=512, nmax=19):
    """Harmonic amplitudes of the weight-3 density coefficient in the fiber angle."""
    e = sp.cancel(sp.together(expr.subs({x: pt[0], y: pt[1]})))
    f = sp.lambdify(m, e, "numpy")
    th = 2 * np.pi * (np.arange(N) + 0.1234) / N
    vals = np.array([float(f(np.tan(t))) / np.cos(t) ** 3 for t in th])
    if not np.all(np.isfinite(vals)):
        return None
    co = np.fft.rfft(vals) / N
    amp = np.abs(co)
    amp[1:] *= 2
    return amp[:nmax + 1]


def part4():
    say("\n=== 4. which term carries the leak")
    r = l2 / l1
    pt = (sp.Rational(3, 7), sp.Rational(-2, 5))
    rng = np.random.default_rng(3)
    A, B = jet(rng), jet(rng)
    F = lift(A, m, vert=sp.Integer(1))
    for rv in (sp.Rational(3, 2), sp.Integer(2)):
        sub = {l1: sp.Integer(1), l2: rv, lam: sp.Integer(1)}
        xp, yp, mp = l1 * x, l2 * y, r * m
        B_at = [q.subs({x: xp, y: yp}, simultaneous=True) for q in B]
        G_img = lift(B_at, mp, vert=sp.Integer(1))
        a, b, c, _ = dtilde(l1 * x, l2 * y)
        LHS = tuple(q.subs(sub) for q in pullback(G_img, a, b, c))
        B11, B12, B22 = B_at
        PB = (l1 ** 2 * B11, l1 * l2 * B12, l2 ** 2 * B22)
        MAIN = tuple(sp.expand(q.subs(sub)) for q in lift(PB, m, vert=sp.Integer(1)))
        Uv2 = (l1 ** 2 + l2 ** 2 * m ** 2) / (1 + m ** 2)
        cc = kappa(PB, m) * ((l1 * l2) ** 2 / Uv2 ** 3 - 1)
        CORR = (sp.together(cc / (1 + m ** 2) ** 2).subs(sub), sp.S(0), sp.S(0))
        full, main, corr = kGz(F, LHS), kGz(F, MAIN), kGz(F, CORR)
        check("r = %s: kGz is linear in its second argument, so full = strained "
              "pair + correction" % rv,
              sp.simplify(sp.together(full - main - corr)) == 0)
        af, am, ac = (spectrum(q, pt) for q in (full, main, corr))
        top = af.max()
        hm = [n for n in range(len(am)) if am[n] > 1e-7 * am.max()]
        say("  r = %-3s  strained pair (II_1, phi^*II_2): harmonics %s" % (rv, hm))
        check("r = %s: the strained pair has harmonics 1, 3, 5 and no others -- it "
              "IS Statement 3.13 of that pair" % rv, hm == [1, 3, 5])
        say("           correction  1,3,5,7,9,11 : %s"
            % np.round([ac[n] for n in (1, 3, 5, 7, 9, 11)], 3))
        say("           full        1,3,5,7,9,11 : %s"
            % np.round([af[n] for n in (1, 3, 5, 7, 9, 11)], 3))
        hi = max(abs(ac[n] - af[n]) for n in range(7, 18, 2))
        check("r = %s: above 5 the correction carries the WHOLE leak (its harmonics "
              ">= 7 are those of the full pairing, to %.0e)" % (rv, hi / top), hi
              < 1e-8 * top)
        check("r = %s: and it contaminates 1, 3 and 5 as well" % rv,
              min(ac[n] for n in (1, 3, 5)) > 1e-6 * top)


# --------------------------------------------------------------------------------- 5

def part5():
    say("\n=== 5. undoing acts on the argument, not on the output")
    r = l2 / l1
    rng = np.random.default_rng(5)
    A, B = jet(rng), jet(rng)
    rv = sp.Rational(3, 2)
    sub = {l1: sp.Integer(1), l2: rv, lam: sp.Integer(1)}
    F = lift(A, m, vert=sp.Integer(1))
    xp, yp, mp = l1 * x, l2 * y, r * m
    B_at = [q.subs({x: xp, y: yp}, simultaneous=True) for q in B]
    G_img = lift(B_at, mp, vert=sp.Integer(1))
    a, b, c, _ = dtilde(l1 * x, l2 * y)
    LHS = tuple(q.subs(sub) for q in pullback(G_img, a, b, c))
    B11, B12, B22 = B_at
    PB = (l1 ** 2 * B11, l1 * l2 * B12, l2 ** 2 * B22)
    MAIN = tuple(sp.expand(q.subs(sub)) for q in lift(PB, m, vert=sp.Integer(1)))
    full, main = kGz(F, LHS), kGz(F, MAIN)
    # the inverse fiber map on the OUTPUT: m |-> m/r  (tan th' = r tan th inverted)
    undone = sp.cancel(sp.together(full.subs(m, m / rv)))
    pt = (sp.Rational(3, 7), sp.Rational(-2, 5))
    au, am = spectrum(undone, pt), spectrum(main, pt)
    check("the inverse fiber map applied to the ANSWER does not recover the "
          "strained-pair pairing", sp.simplify(sp.together(undone - main)) != 0)
    if au is not None:
        hi = max(au[n] for n in range(7, 20, 2))
        say("  after the inverse map the harmonics >= 7 are still %.3f of the "
            "largest" % (hi / au.max()))
        check("and the leaked harmonics are still there afterwards",
              hi > 1e-6 * au.max())
    check("whereas the argument that has no leak is Lambda_{g_1}(phi^*II_2): its "
          "pairing has harmonics 1, 3, 5 alone",
          [n for n in range(len(am)) if am[n] > 1e-7 * am.max()] == [1, 3, 5])
    # kGz is bilinear, and only one argument was moved
    G1c, G2c = jet(np.random.default_rng(31)), jet(np.random.default_rng(32))
    L1 = lift(G1c, m, vert=sp.Integer(1))
    L2 = lift(G2c, m, vert=sp.Integer(1))
    s_ = sp.Symbol("s_")
    mix = tuple(p + s_ * q for p, q in zip(L1, L2))
    check("kGz is linear in its second argument (whence the decomposition of the "
          "pairing follows from that of the lift)",
          sp.simplify(sp.expand(kGz(F, mix) - kGz(F, L1) - s_ * kGz(F, L2))) == 0)


# --------------------------------------------------------------------------------- 6

def part6():
    say("\n=== 6. a stretch that varies: what is new")
    rng = np.random.default_rng(13)
    B = jet(rng)
    eps = sp.Symbol("eps")
    # phi = (l_1(x) x, l_2 y) with l_1 varying: d phi is no longer constant
    phi1, phi2 = l1 * x + eps * x ** 2 / 2, l2 * y
    a, b, c, M = dtilde(phi1, phi2)
    check("for a non-affine phi the mixed coefficient c = e_2(M) is non-zero, and "
          "it vanishes with the derivative of d phi", sp.simplify(c.subs(eps, 0)) == 0
          and sp.simplify(c) != 0)
    B_at = [q.subs({x: phi1, y: phi2}, simultaneous=True) for q in B]
    G_img = lift(B_at, M)
    F2, F1, F0 = pullback(G_img, a, b, c)
    check("the pulled-back lift then has a MIXED component, F_1 = a(c G_2 + b G_1), "
          "which is the only genuinely new object", sp.simplify(F1.subs(eps, 0)) == 0
          and sp.simplify(F1) != 0)
    check("F_1 is proportional to c, i.e. to the derivative of the stretch",
          sp.simplify(sp.cancel(sp.together(F1 - a * c * G_img[0]))) == 0)
    # and the horizontal component picks up a term of SECOND order in c.  Read at
    # exact rational points: the full symbolic `simplify` of this one rational
    # function costs more than the whole of the rest of the script.
    pts = [{x: sp.Rational(3, 7), y: sp.Rational(-2, 5), m: sp.Rational(2, 9),
            l1: sp.Integer(2), l2: sp.Integer(3), eps: sp.Rational(1, 2),
            lam: sp.Integer(1)},
           {x: sp.Rational(-1, 3), y: sp.Rational(4, 5), m: sp.Rational(-5, 4),
            l1: sp.Integer(1), l2: sp.Rational(3, 2), eps: sp.Rational(-2, 3),
            lam: sp.Rational(5, 7)},
           {x: sp.Rational(1, 5), y: sp.Rational(1, 6), m: sp.Rational(7, 3),
            l1: sp.Rational(4, 3), l2: sp.Integer(1), eps: sp.Integer(2),
            lam: sp.Integer(-3)}]
    ok = []
    for P in pts:
        extra = sp.cancel(F0.subs(P) - (b ** 2 * G_img[2]).subs(P))
        ok.append(sp.simplify(extra - sp.cancel((c ** 2 * G_img[0]).subs(P))) == 0
                  and extra != 0)
    check("the horizontal component picks up c^2 G_2 as well -- second order in the "
          "strain derivative, so not a new object but a second new term (%d exact "
          "points)" % len(pts), all(ok))
    say("  so the variable-stretch decomposition has, beyond the two constant-stretch "
        "terms,\n  one mixed term of first order and one horizontal term of second "
        "order in grad U.")


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "leak.log"), "w")
    say("How the leak of Statement 5.5 decomposes, and what undoes it.")
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    say("\n%d checks, %d passed, %d failed"
        % (CHECKS[0], CHECKS[1], CHECKS[0] - CHECKS[1]))
    LOG.close()


if __name__ == "__main__":
    main()
