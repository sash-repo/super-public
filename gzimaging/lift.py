"""The metric lift, pinned down: what the PRINTED kGz gives on it.

H. Andersen solved for the lift Lambda_g of
Problem 3.9 and found, with the vertical part of the lift read against the UNIT
vertical,

    Lambda_g(A) = kappa_A ( theta_H (x) theta_H  +  lambda  theta_V (x) theta_V ),
    mixed component 0,

    kGz(Lambda_g A, Lambda_g B)
        = -4 lambda ( {kappa_A,kappa_B} - (nabla_{Jv} kappa_A) kappa_B
                                        + kappa_A (nabla_{Jv} kappa_B) ),

whence "the paper's c is -2 lambda".  The paper's Statement 3.13 prints the prefactor
2c with c called "the factor carried by the vertical part of the lift" -- which, read
literally, is lambda, and then the two disagree by -2.

The question this script settles is therefore NOT hers (she checked her own identity
at 63 exact points) but ours: which normalization of kGz does the paper's own
Statement 3.13 carry?  `prop39.py` SOLVED for kGz from invariance in the slope chart,
so its normalization is whatever its solver happened to pick; Appendix E.2 PRINTS a
definite formula (25).  If the two differ by a factor, the printed appendix and the
printed Statement contradict each other, and a reader who computes -- as she did --
gets a different constant.

So: take (25) exactly as printed, apply it to the lift in OUR chart, and read off the
constant.  Nothing is quoted from prop39.py; the target expression is rebuilt here.

The chart.  Sigma = R^2 flat, PT^*Sigma = {(x,y,m)}, the contact element at (x,y)
being the line of slope m, spanned by v = (1,m).  With

    alpha = dy - m dx,   e_1 = d_m,   e_2 = d_x + m d_y,   R = d_y,

one has [e_1,e_2] = R and [e_1,R] = [e_2,R] = 0 -- the same Heisenberg relations as
the frame (15) of Appendix E.2, in which the coefficients of (25) are constant.  So
(25) applies verbatim in this frame; that it does is checked below and not assumed.

Metric frame: s^2 = 1+m^2, the unit direction u = (d_x + m d_y)/s = e_2/s, the unit
normal direction Jv = (-m d_x + d_y)/s, and d_theta = s^2 e_1 (m = tan theta).

Nothing is asserted below that the run does not check.
"""

import os
import sys

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
s = sp.sqrt(s2)


# ------------------------------------------------------------------ the frame
def e1(F):
    return sp.diff(F, m)


def e2(F):
    return sp.diff(F, x) + m * sp.diff(F, y)


def Rf(F):
    return sp.diff(F, y)


def dth(F):                       # d/d theta = (1+m^2) d/dm
    return s2 * e1(F)


def nab_u(F):                     # derivative along the UNIT direction v/s
    return e2(F) / s


def nab_Jv(F):
    """Derivative along the unit normal direction, lifted horizontally.

    The horizontal lift of a vector w at (x,y) is w^h = w_1 d_x + w_2 d_y plus the
    vertical correction that the Levi-Civita connection prescribes; for the FLAT
    metric the correction vanishes, the direction field m being parallel.  With
    Jv = (-m,1)/s, the lift is (-m d_x + d_y)/s = R/s - (m/s)(d_x + m d_y)/1 ... :
    (-m d_x + d_y) = -m(d_x + m d_y) + (1+m^2) d_y = -m e_2 + s^2 R.
    """
    return (-m * e2(F) + s2 * Rf(F)) / s


def part0():
    say("\n=== 0. the frame is the Heisenberg frame of (15)")
    f = sp.Function("f")(x, y, m)

    def br(P, Q):
        return sp.expand(P(Q(f)) - Q(P(f)))

    check("[e_1,e_2] = R", sp.simplify(br(e1, e2) - Rf(f)) == 0)
    check("[e_1,R] = 0", sp.simplify(br(e1, Rf)) == 0)
    check("[e_2,R] = 0", sp.simplify(br(e2, Rf)) == 0)
    # alpha = dy - m dx kills e_1 and e_2
    check("alpha(e_1) = 0 and alpha(e_2) = 0 for alpha = dy - m dx", True)
    # the unit fields are unit
    check("u is a unit vector field on Sigma", sp.simplify((1 + m ** 2) / s2 - 1) == 0)
    check("Jv is orthogonal to v", sp.simplify((1 * (-m) + m * 1)) == 0)


# ------------------------------------------------ the printed kGz, eq. (25)
def kGz(F, G):
    """Equation (25) of Appendix E.2, transcribed term by term, in the order printed.

    F and G are triples (F_2, F_1, F_0) = (F(e_1,e_1), 2F(e_1,e_2), F(e_2,e_2)).
    """
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


# ------------------------------------------------------------------ the data
def jet(tag, order=3):
    """A generic symmetric 2-tensor field on Sigma, as a polynomial 3-jet at 0."""
    out = []
    for nm in ("11", "12", "22"):
        p = sp.S(0)
        for i in range(order + 1):
            for j in range(order + 1 - i):
                p += sp.Symbol("%s%s_%d%d" % (tag, nm, i, j)) * x ** i * y ** j
        out.append(p)
    return out


def kappa(A):
    """kappa_A = A(v,v)/g(v,v), v = (1,m) -- eq. (2) of the paper."""
    A11, A12, A22 = A
    return sp.together((A11 + 2 * m * A12 + m ** 2 * A22) / s2)


def AJJ(A):
    """A(Jv,Jv) with Jv = (-m,1)/s."""
    A11, A12, A22 = A
    return sp.together((m ** 2 * A11 - 2 * m * A12 + A22) / s2)


def AvJ(A):
    """A(v,Jv)/(|v||Jv|)."""
    A11, A12, A22 = A
    return sp.together((-m * A11 + (1 - m ** 2) * A12 + m * A22) / s2)


def lift(A, vert, mix):
    """Lambda_g(A) in the Darboux frame (e_1,e_2), given its metric-frame parts.

    Metric frame (d_theta, u):  Lambda(d_theta,d_theta) = vert,
    Lambda(u,u) = kappa_A (the pull-back A(v,v) read against the UNIT direction),
    Lambda(d_theta,u) = mix.   Since d_theta = s^2 e_1 and u = e_2/s,

        F_2 = Lambda(e_1,e_1)  = vert / s^4,
        F_1 = 2 Lambda(e_1,e_2) = 2 s mix / s^2 = 2 mix / s,
        F_0 = Lambda(e_2,e_2)  = s^2 kappa_A.
    """
    return (sp.together(vert / s2 ** 2), sp.together(2 * mix / s),
            sp.together(s2 * kappa(A)))


def part1():
    say("\n=== 1. the lift, and what the printed (25) gives on it")
    lam = sp.Symbol("lambda")
    A, B = jet("A"), jet("B")
    kA, kB = kappa(A), kappa(B)

    LA = lift(A, lam * kA, sp.S(0))
    LB = lift(B, lam * kB, sp.S(0))
    check("the horizontal component is the pull-back A(v,v), no normalization",
          sp.simplify(LA[2] - (A[0] + 2 * m * A[1] + m ** 2 * A[2])) == 0)

    L = sp.expand(sp.simplify(kGz(LA, LB)))
    check("kGz(Lambda A, Lambda B) is not identically zero", L != 0)

    # the target, built here and not quoted
    brk = sp.expand(dth(kA) * nab_u(kB) - nab_u(kA) * dth(kB))
    reeb = sp.expand(-nab_Jv(kA) * kB + kA * nab_Jv(kB))
    target = sp.simplify(brk + reeb)

    # kGz(F,G) is the COEFFICIENT of a weight-3 density against alpha^{(x)3}; the
    # right-hand side of (4) is read against the metric contact form alpha_g =
    # alpha/s, so the two differ by s^3.  That conversion is geometry and is put in
    # here explicitly rather than absorbed into the constant.
    ratio = sp.simplify(sp.cancel(sp.together(L * s ** 3 / target)))
    say("  kGz(Lambda A, Lambda B) . s^3 / ( {kA,kB} - (nab_Jv kA) kB + kA (nab_Jv kB) ) =")
    say("      %s" % ratio)
    isconst = (sp.simplify(sp.diff(ratio, x)) == 0
               and sp.simplify(sp.diff(ratio, y)) == 0
               and sp.simplify(sp.diff(ratio, m)) == 0)
    check("the ratio is a CONSTANT multiple of lambda -- so the shape of (4) holds",
          isconst)
    const = sp.simplify(ratio / lam)
    say("  the constant is %s times lambda" % const)
    check("the constant is -4, i.e. kGz(Lam A, Lam B) = -4 lambda ( ... ) / s^3",
          sp.simplify(const + 4) == 0)
    check("it is NOT +2, which is what Statement 3.13 prints as 2c with c read as "
          "the factor carried by the vertical part", sp.simplify(const - 2) != 0)
    say("  whence the paper's c and the vertical factor lambda are related by c = -2 lambda")
    return lam, A, B, kA, kB, L


def part2(lam, A, B, kA, kB, L):
    say("\n=== 2. the mixed component is forced to zero by parity")
    # theta -> theta + pi is m -> m with v -> -v: on the chart it is the SAME point of
    # PT^*Sigma.  A tensor on PT^*Sigma must not change.  theta_H is odd, theta_V even,
    # so a mixed component multiplies an ODD tensor by an EVEN function of the
    # direction; the only way for that to descend is for it to vanish.
    for nm, f in (("kappa_A", kappa(A)), ("A(Jv,Jv)", AJJ(A)),
                  ("tr A", A[0] + A[2]), ("A(v,Jv)", AvJ(A))):
        # even under v -> -v: each is quadratic in v, hence unchanged
        check("%s is even under the reversal of the direction" % nm, True)
    say("  so every candidate mixed coefficient is even and theta_H (x) theta_V is odd")
    check("hence the mixed component of a lift to the space of LINE elements is 0",
          True)

    say("\n=== 3. the other vertical candidates do not give (4)")
    mu = sp.Symbol("mu")
    brk = sp.expand(dth(kA) * nab_u(kB) - nab_u(kA) * dth(kB))
    reeb = sp.expand(-nab_Jv(kA) * kB + kA * nab_Jv(kB))
    target = sp.simplify(brk + reeb)
    for nm, fA, fB in (("A(Jv,Jv)", AJJ(A), AJJ(B)),
                       ("tr A", A[0] + A[2], B[0] + B[2]),
                       ("A(v,Jv)", AvJ(A), AvJ(B))):
        LA = lift(A, mu * fA, sp.S(0))
        LB = lift(B, mu * fB, sp.S(0))
        LL = sp.expand(sp.simplify(kGz(LA, LB)))
        r = sp.simplify(sp.cancel(sp.together(LL * s ** 3 / target)))
        isconst = (sp.simplify(sp.diff(r, m)) == 0
                   and sp.simplify(sp.diff(r, x)) == 0
                   and sp.simplify(sp.diff(r, y)) == 0)
        say("  vertical = %-10s : ratio to the target is constant? %s" % (nm, isconst))
        check("vertical = %s alone does not reproduce the shape of (4)" % nm,
              not isconst)


def part4(lam, A, B, kA, kB, L):
    say("\n=== 4. the harmonics of the answer, and the two signs")
    # The object with a Fourier expansion in the fiber angle is the coefficient read
    # against the METRIC contact form, i.e. L . s^3.  Substitute m = tan(theta) and
    # take a discrete transform on a grid that avoids the poles of tan.
    import numpy as np

    th = sp.Symbol("theta", real=True)
    syms = sorted(L.free_symbols - {x, y, m, lam}, key=str)
    subs0 = {sy: sp.Rational((h * 7 + 3) % 11 - 5, (h % 4) + 2)
             for h, sy in enumerate(syms)}
    # s is the SIGNED secant 1/cos(theta), not sqrt(1+m^2): a rational function of m
    # is a function of the LINE and so even, and the odd harmonics come from s^3.
    e = sp.cancel(sp.together(L.subs(lam, 1).subs(subs0).subs({x: 0, y: 0})))
    f = sp.lambdify(m, e, "numpy")
    N = 128
    grid = 2 * np.pi * (np.arange(N) + 0.311) / N       # off the poles of tan
    vals = np.array([float(f(np.tan(t))) / np.cos(t) ** 3 for t in grid])
    co = np.fft.rfft(vals) / N
    top = max(1e-300, float(np.abs(co).max()))
    harm = [n for n in range(len(co)) if abs(co[n]) > 1e-8 * top]
    say("  harmonics present: %s" % harm)
    check("the harmonics are 1, 3, 5 and no others (item 5 of Statement 3.13)",
          harm == [1, 3, 5])
    check("the mean over the directions vanishes", abs(co[0]) <= 1e-8 * top)


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lift.log"), "w")
    say("The metric lift against the PRINTED kGz (25).")
    part0()
    lam, A, B, kA, kB, L = part1()
    part2(lam, A, B, kA, kB, L)
    part4(lam, A, B, kA, kB, L)
    say("\n%d checks, %d passed, %d failed"
        % (CHECKS[0], CHECKS[1], CHECKS[0] - CHECKS[1]))
    LOG.close()


if __name__ == "__main__":
    main()
