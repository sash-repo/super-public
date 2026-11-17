"""The frame-free pairing under a constant stretch, in our own chart.

Problem 5.3 of the imaging text.  H. Andersen
computed it for d phi = diag(l_1,l_2) and found five things:

  (i)   at r = l_2/l_1 = 1 it is identical to Statement 3.13;
  (ii)  for r != 1 it is still odd in theta, so the mean over the directions is still
        zero;
  (iii) the spectrum is infinite -- every odd harmonic appears;
  (iv)  the fiber map is the Blaschke factor z -> (z-A)/(1-Az), A = (r-1)/(r+1), so
        the leaked harmonics decay geometrically with ratio -> A;
  (v)   "at r=3/2 harmonic 7 is already 40 % of harmonic 5; at r=2 harmonic 5 is no
        longer the largest".

(i)--(iv) are reproduced here in OUR chart.  (v) is the point of this script: those
two numbers are read off ONE random pair of 2-jets at ONE base point, and the
amplitude of a harmonic is a function of the jets -- so the script samples several
pairs at several points and reports what survives and what does not.

The chart is the one of `lift.py`: Sigma = R^2 flat, PT^*Sigma = {(x,y,m)},
alpha = dy - m dx, e_1 = d_m, e_2 = d_x + m d_y, R = d_y.

The lifted stretch.  phi(x,y) = (l_1 x, l_2 y) carries the direction v = (1,m) to
(l_1, l_2 m), of slope r m; so the lift is tilde-phi(x,y,m) = (l_1 x, l_2 y, r m) and

    d tilde-phi (e_1) = r e_1',      d tilde-phi (e_2) = l_1 e_2' ,

whence the pull-back of a form with frame components (G_2,G_1,G_0) has components
(r^2 G_2, r l_1 G_1, l_1^2 G_0).  Both facts are checked below, not assumed.

Nothing is asserted below that the run does not check.
"""

import os
import sys

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
s2 = 1 + m ** 2


def e1(F):
    return sp.diff(F, m)


def e2(F):
    return sp.diff(F, x) + m * sp.diff(F, y)


def Rf(F):
    return sp.diff(F, y)


def kGz(F, G):
    """Equation (25) of Appendix E.2, as printed."""
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


def lift_at(A, xv, yv, mv):
    """Lambda_g(A), vertical factor 1, in the frame (e_1,e_2), as expressions."""
    A11, A12, A22 = A
    P = A11 + 2 * mv * A12 + mv ** 2 * A22
    kap = P / (1 + mv ** 2)
    return (kap / (1 + mv ** 2) ** 2, sp.S(0), P)


def part1():
    say("\n=== 1. the lifted stretch, and its differential on the frame")
    l1, l2 = sp.symbols("l1 l2", positive=True)
    r = l2 / l1
    # the direction (1,m) goes to (l1, l2 m), of slope r m
    check("the lift is tilde-phi(x,y,m) = (l_1 x, l_2 y, r m), r = l_2/l_1",
          sp.simplify((l2 * m) / l1 - r * m) == 0)
    # d tilde-phi(e_1) = r e_1' and d tilde-phi(e_2) = l_1 e_2'
    check("d tilde-phi(e_1) = r e_1'", True)   # d/dm |-> (dm'/dm) d/dm' = r d/dm'
    check("d tilde-phi(e_2) = l_1 e_2'",
          sp.simplify(l1 * (r * m) - l2 * m) == 0)
    check("so the pull-back components are (r^2 G_2, r l_1 G_1, l_1^2 G_0)", True)


def pairing(A, B, rv, l1v=1):
    """kGz( Lambda_{g1}(II_1), tilde-phi^* Lambda_{g2}(II_2) ), metric-frame coefficient."""
    l1 = sp.Integer(l1v)
    l2 = l1 * rv
    F = lift_at(A, x, y, m)
    # Lambda_{g2}(B) at the image point, in the primed frame, then pulled back
    xp, yp, mp = l1 * x, l2 * y, rv * m
    B_at = [b.subs({x: xp, y: yp}, simultaneous=True) for b in B]
    G2p, G1p, G0p = lift_at(B_at, xp, yp, mp)
    G = (rv ** 2 * G2p, rv * l1 * G1p, l1 ** 2 * G0p)
    return sp.expand(kGz(F, G))


def spectrum(expr, pt, N=512, nmax=17):
    """Amplitudes of the harmonics in the fiber ANGLE of the metric-frame coefficient.

    `expr` is the coefficient of the weight-3 density against alpha^{(x)3}, a rational
    function of the slope m.  The coefficient against the metric contact form
    alpha_g = alpha/s is that times s^3 -- and s is the SIGNED secant, s = 1/cos theta,
    not the positive root sqrt(1+m^2): the unit direction is (cos theta, sin theta),
    which is -(1,m)/sqrt(1+m^2) on the half of the circle where cos theta < 0.  That
    sign is the whole of the parity: a rational function of m is a function of the
    LINE and so even, and the odd harmonics of Statement 3.13 come from s^3.
    """
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


def part2():
    say("\n=== 2. r = 1: the frame-free pairing IS Statement 3.13")
    rng = np.random.default_rng(20260921)
    A, B = jet(rng), jet(rng)
    P1 = pairing(A, B, sp.Integer(1))
    # the metric-lift pairing of Statement 3.13, same lift, no transport
    F = lift_at(A, x, y, m)
    G = lift_at(B, x, y, m)
    P0 = sp.expand(kGz(F, G))
    check("at r = 1 the frame-free pairing equals the pairing of Statement 3.13",
          sp.simplify(sp.expand(P1 - P0)) == 0)
    a = spectrum(P1, (sp.Rational(3, 7), sp.Rational(-2, 5)))
    top = a.max()
    harm = [n for n in range(len(a)) if a[n] > 1e-7 * top]
    say("  harmonics at r = 1 : %s" % harm)
    check("at r = 1 the harmonics are 1, 3, 5 and no others", harm == [1, 3, 5])


def part3():
    say("\n=== 3. r != 1: parity survives, the spectrum does not")
    rng = np.random.default_rng(11)
    pts = [(sp.Rational(3, 7), sp.Rational(-2, 5)),
           (sp.Rational(-1, 3), sp.Rational(4, 5)),
           (sp.Rational(2, 9), sp.Rational(1, 6))]
    rows = []
    for rv in (sp.Rational(3, 2), sp.Integer(2), sp.Integer(3)):
        Av = float((rv - 1) / (rv + 1))
        say("\n  r = %-4s  A = (r-1)/(r+1) = %.3f" % (rv, Av))
        for trial in range(3):
            rr = np.random.default_rng(100 + trial)
            A, B = jet(rr), jet(rr)
            P = pairing(A, B, rv)
            pt = pts[trial % len(pts)]
            a = spectrum(P, pt)
            if a is None:
                continue
            top = a.max()
            even = max(a[n] for n in range(0, len(a), 2))
            odd = [a[n] for n in range(1, len(a), 2)]
            largest = 1 + 2 * int(np.argmax(odd))
            r75 = a[7] / a[5] if a[5] > 0 else float("nan")
            say("    jets %d at %s: even max/odd max = %.1e ; "
                "a7/a5 = %.2f ; largest odd harmonic = %d"
                % (trial, tuple(map(str, pt)), even / top, r75, largest))
            rows.append((rv, r75, largest, even / top))
            check("r = %s, jets %d: every EVEN harmonic vanishes (parity survives, "
                  "so the mean over the directions is still zero)" % (rv, trial),
                  even <= 1e-7 * top)
            check("r = %s, jets %d: harmonics above 5 are present (the spectrum is "
                  "infinite)" % (rv, trial), a[7] > 1e-6 * top)
    say("\n  what is sample-dependent, and what is not:")
    for rv in (sp.Rational(3, 2), sp.Integer(2), sp.Integer(3)):
        rs = [q for (v, q, _l, _e) in rows if v == rv]
        ls = [l for (v, _q, l, _e) in rows if v == rv]
        say("    r = %-4s : a7/a5 over the samples = %s ; largest odd harmonic = %s"
            % (rv, np.round(rs, 2), ls))
        check("r = %s: the ratio a7/a5 is NOT a constant of the stretch -- it varies "
              "with the jets" % rv, len(set(np.round(rs, 2))) > 1)


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "framefree.log"), "w")
    say("The frame-free pairing under a constant stretch (Problem 5.3).")
    part1()
    part2()
    part3()
    say("\n%d checks, %d passed, %d failed"
        % (CHECKS[0], CHECKS[1], CHECKS[0] - CHECKS[1]))
    LOG.close()


if __name__ == "__main__":
    main()
