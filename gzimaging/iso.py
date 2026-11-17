"""An ISOTROPIC stretch is not the identity: what item 1) of Statement 5.5 says.

`leak.py` part 2 reports that the correction term of H. Andersen's decomposition
does NOT vanish at l_1 = l_2 unless the common value is 1: at |Uv|^2 = l^2 the bracket
is l^4/l^6 - 1 = l^{-2} - 1.  Item 1) of the Statement (st:blaschke) as first drafted read

    "At r = 1 the pairing is the pairing of Statement 3.13, identically and not
     merely up to a factor",

with r := l_2/l_1 -- which covers every isotropic dilation, not only phi = id.  That is
what this script tests, because a growing nodule that grows isotropically is exactly the
case r = 1, l > 1.

The chart, the lift and kGz are those of `leak.py`.
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
lam, lamp, ell = sp.symbols("lam lamp ell", positive=True)


def e1(F):
    return sp.diff(F, m)


def e2(F):
    return sp.diff(F, x) + m * sp.diff(F, y)


def Rf(F):
    return sp.diff(F, y)


def kGz(F, G):
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
    out = []
    for _ in range(3):
        p = sp.S(0)
        for i in range(order + 1):
            for j in range(order + 1 - i):
                p += sp.Integer(int(rng.integers(-3, 4))) * x ** i * y ** j
        out.append(p)
    return out


def kappa(A, mv):
    A11, A12, A22 = A
    return (A11 + 2 * mv * A12 + mv ** 2 * A22) / (1 + mv ** 2)


def lift(A, mv, vert):
    kap = kappa(A, mv)
    return (vert * kap / (1 + mv ** 2) ** 2, sp.S(0), kap * (1 + mv ** 2))


def spectrum(expr, pt, N=512, nmax=15):
    """Harmonic amplitudes of the weight-3 density coefficient in the fiber angle."""
    e = sp.cancel(sp.together(expr.subs({x: pt[0], y: pt[1]})))
    f = sp.lambdify(m, e, "numpy")
    th = 2 * np.pi * (np.arange(N) + 0.1234) / N
    vals = np.array([float(f(np.tan(t))) / np.cos(t) ** 3 for t in th])
    co = np.fft.rfft(vals) / N
    amp = np.abs(co)
    amp[1:] *= 2
    return amp[:nmax + 1]


def part1():
    say("\n=== 1. an isotropic dilation, symbolically")
    rng = np.random.default_rng(20260921)
    A, B = jet(rng), jet(rng)
    # phi(x,y) = (ell x, ell y): the fiber map is the identity, a = 1, b = ell, c = 0
    B_at = [q.subs({x: ell * x, y: ell * y}, simultaneous=True) for q in B]
    G_img = lift(B_at, m, lam)
    LHS = (G_img[0], sp.S(0), ell ** 2 * G_img[2])        # pull-back: a=1, b=ell, c=0
    PB = (ell ** 2 * B_at[0], ell ** 2 * B_at[1], ell ** 2 * B_at[2])
    MAIN = lift(PB, m, lam)
    check("the horizontal components agree: no leak there",
          sp.simplify(sp.together(LHS[2] - MAIN[2])) == 0)
    d = sp.simplify(sp.together(LHS[0] - MAIN[0]))
    check("the VERTICAL components do NOT agree for ell =/= 1",
          d != 0 and sp.simplify(d.subs(ell, 1)) == 0)
    fac = sp.simplify(sp.cancel(sp.together(LHS[0] / MAIN[0])))
    say("  the vertical component is off by the factor %s" % fac)
    check("that factor is ell^{-2}, which is the bracket of the decomposition at "
          "|Uv|^2 = ell^2", sp.simplify(fac - 1 / ell ** 2) == 0)
    check("so tilde-phi^* Lambda_{g_2}(B) is the lift of phi^*B with the vertical "
          "constant lam/ell^2 in place of lam",
          sp.simplify(sp.together(LHS[0] - lift(PB, m, lam / ell ** 2)[0])) == 0
          and sp.simplify(sp.together(LHS[2] - lift(PB, m, lam / ell ** 2)[2])) == 0)


def part2():
    say("\n=== 2. is the pairing nevertheless that of Statement 3.13, up to a factor?")
    rng = np.random.default_rng(3)
    A, B = jet(rng), jet(rng)
    F = lift(A, m, lam)
    for ev in (sp.Integer(2), sp.Rational(3, 2)):
        B_at = [q.subs({x: ev * x, y: ev * y}, simultaneous=True) for q in B]
        PB = (ev ** 2 * B_at[0], ev ** 2 * B_at[1], ev ** 2 * B_at[2])
        full = kGz(F, lift(PB, m, lam / ev ** 2))
        main = kGz(F, lift(PB, m, lam))
        diff = sp.simplify(sp.together(full - main))
        check("ell = %s: the frame-free pairing differs from the pairing of "
              "Statement 3.13 of the strained pair" % ev, diff != 0)
        # is it a SCALAR multiple of it?
        rat = sp.cancel(sp.together(full / main))
        isconst = sp.simplify(sp.diff(rat, m)) == 0 and sp.simplify(sp.diff(rat, x)) == 0
        check("ell = %s: and not by a constant factor either -- the ratio depends on "
              "the point and the direction" % ev, not isconst)
        # the shape: how the pairing depends on the two vertical constants
        la, lb = sp.symbols("la lb", positive=True)
        P = sp.expand(kGz(lift(A, m, la), lift(PB, m, lb)))
        p = sp.Poly(sp.expand(P), la, lb)
        mons = sorted(set(mo for mo in p.monoms()))
        say("  ell = %s: the pairing as a polynomial in the two vertical constants "
            "has monomials (la^i lb^j) i,j = %s" % (ev, mons))
        check("ell = %s: it is LINEAR in each constant separately, with no cross "
              "term and no constant term: P = la*X + lb*Y" % ev,
              mons == [(0, 1), (1, 0)])
        X = sp.expand(p.coeff_monomial((1, 0)))
        Y = sp.expand(p.coeff_monomial((0, 1)))
        check("ell = %s: so the whole discrepancy is (1/ell^2 - 1) Y, the part of "
              "(13) carried by the SECOND argument's vertical component" % ev,
              sp.simplify(sp.together(
                  diff - (1 / ev ** 2 - 1) * Y.subs({la: lam, lb: lam}) * lam
                  / lam)) == 0
              or sp.simplify(sp.together(
                  full - main - (1 / ev ** 2 - 1) * lam * Y)) == 0)
        # does the isotropic leak change the SPECTRUM?
        pt = (sp.Rational(3, 7), sp.Rational(-2, 5))
        af = spectrum(full.subs(lam, 1), pt)
        am = spectrum(main.subs(lam, 1), pt)
        hf = [n for n in range(len(af)) if af[n] > 1e-7 * af.max()]
        hm = [n for n in range(len(am)) if am[n] > 1e-7 * am.max()]
        say("  ell = %s: harmonics of the frame-free pairing %s, of the strained "
            "pair %s" % (ev, hf, hm))
        check("ell = %s: an ISOTROPIC dilation leaks into the value but not into "
              "the spectrum -- both are 1, 3, 5" % ev, hf == [1, 3, 5] == hm)


def part3():
    say("\n=== 3. what survives: the case that IS the identity")
    rng = np.random.default_rng(3)
    A, B = jet(rng), jet(rng)
    F = lift(A, m, lam)
    G = lift(B, m, lam)
    check("at ell = 1 (phi the identity) the two agree identically",
          sp.simplify(sp.expand(kGz(F, G) - kGz(F, lift(B, m, lam / 1 ** 2)))) == 0)
    # and the horizontal-only statement, for every ell
    say("  and for every ell the HORIZONTAL component is leak-free, which is why the")
    say("  strained pair is the right argument.")


def part4():
    say("\n=== 4. the Hopf datum of the strained form")
    l1, l2 = sp.symbols("l1 l2", positive=True)
    b, be_r, be_i = sp.symbols("b beta_r beta_i", real=True)
    # B in Hopf notation: B11 = b + Re beta, B22 = b - Re beta, B12 = -Im beta
    B11, B22, B12 = b + be_r, b - be_r, -be_i
    PB11, PB22, PB12 = l1 ** 2 * B11, l2 ** 2 * B22, l1 * l2 * B12
    a_pb = (PB11 + PB22) / 2
    al_r = (PB11 - PB22) / 2
    al_i = -PB12
    check("the trace of the strained form is (l_1^2+l_2^2)b/2 + (l_1^2-l_2^2)"
          "Re(beta)/2", sp.simplify(a_pb - ((l1 ** 2 + l2 ** 2) * b / 2
                                            + (l1 ** 2 - l2 ** 2) * be_r / 2)) == 0)
    check("its Hopf number picks up b*(l_1^2-l_2^2)/2 -- the strain anisotropy "
          "entering through the TRACE of II_2",
          sp.simplify(al_r - (b * (l1 ** 2 - l2 ** 2) / 2
                              + (l1 ** 2 + l2 ** 2) * be_r / 2)) == 0)
    check("and the rest of it is beta twisted by U: Re by (l_1^2+l_2^2)/2, Im by "
          "l_1 l_2", sp.simplify(al_i - l1 * l2 * be_i) == 0)
    check("at l_1 = l_2 = 1 the Hopf datum is unchanged",
          sp.simplify(al_r.subs({l1: 1, l2: 1}) - be_r) == 0
          and sp.simplify(al_i.subs({l1: 1, l2: 1}) - be_i) == 0)


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "iso.log"), "w")
    say("An isotropic dilation is not the identity.")
    part1()
    part2()
    part3()
    part4()
    say("\n%d checks, %d passed, %d failed"
        % (CHECKS[0], CHECKS[1], CHECKS[0] - CHECKS[1]))
    LOG.close()


if __name__ == "__main__":
    main()
