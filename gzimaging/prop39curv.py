"""Problem 3.9 with the curvature in: the obstacle it names is not an obstacle.

Problem 3.9 says the computation "needs the Levi-Civita connection inside the frame of
D, so the engine of [LeLoGz, Part II], which works with a flat model, has to carry
curvature terms first".  It has not: the contact structure of PT^*Sigma is canonical
and knows nothing of g, so in a Darboux chart of it -- which exists whatever g is --
the operator kGz has the SAME constant coefficients as in the flat model.  The metric
enters the problem in one place only, the lift Lam_g, i.e. the ARGUMENT.

So the computation is: build Lam_g for a general g in the canonical Darboux frame of
PT^*R^2, and apply the operator prop39.py has already solved for.  Carried out here at
a point, in normal coordinates, to first order in the Gauss curvature K -- which is
all there is, an expression of weight 2 having room for K and for nothing else built
from g (a nabla K would be of weight 3).

  g_ij = delta_ij - (K/3)(delta_ij r^2 - x_i x_j) + O(r^3),   r^2 = x^2+y^2,

whose Gauss curvature at the origin is K -- verified below, not quoted.

Run:  python3 prop39curv.py
"""

import os
import sys

import sympy as sp

import prop39 as P
from prop39 import x, y, m, KEYS, e1, e2, Rd, build_op, jets, bracket, harmonics

LOG = None
CHECKS = [0, 0]


def say(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def check(name, ok):
    CHECKS[0] += 1
    CHECKS[1] += bool(ok)
    say(("  PASS  " if ok else "  FAIL  ") + name)
    return bool(ok)


K = sp.Symbol("K")
ORD = 3                       # truncation order in (x,y); K enters linearly


_EP = sp.Symbol("epsilon", positive=True)


def trunc(e, ord_=ORD):
    """Drop K^2 and the orders in (x,y) beyond what a 2-jet at the origin sees.

    Done by series, not by sifting monomials: the expressions carry square roots of
    the metric, so "degree in (x,y)" is read off a scaling x -> eps x, y -> eps y."""
    e = sp.sympify(e)
    if K in e.free_symbols:
        e = sp.expand(sp.series(e, K, 0, 2).removeO())
    if e.free_symbols & {x, y}:
        e = sp.series(sp.expand(e.subs({x: _EP * x, y: _EP * y})),
                      _EP, 0, ord_ + 1).removeO().subs(_EP, 1)
    return sp.expand(e)


def metric():
    """Normal coordinates at the origin."""
    r2 = x ** 2 + y ** 2
    X = [x, y]
    g = sp.Matrix(2, 2, lambda i, j: sp.KroneckerDelta(i, j)
                  - K * (sp.KroneckerDelta(i, j) * r2 - X[i] * X[j]) / 3)
    return g


def christoffel(g):
    X = [x, y]
    gi = sp.Matrix(2, 2, lambda i, j: trunc(sp.expand(
        g.inv()[i, j].series(K, 0, 2).removeO())))
    G = [[[0] * 2 for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                G[i][j][k] = trunc(sum(
                    gi[i, l] * (sp.diff(g[l, j], X[k]) + sp.diff(g[l, k], X[j])
                                - sp.diff(g[j, k], X[l])) for l in range(2)) / 2)
    return gi, G


def gauss(g, G):
    """K = R_{1212}/det g, computed from the Christoffels."""
    X = [x, y]
    def Riem(i, j, k, l):                      # R^i_{jkl}
        return sp.expand(sp.diff(G[i][j][l], X[k]) - sp.diff(G[i][j][k], X[l])
                         + sum(G[i][k][s] * G[s][j][l] - G[i][l][s] * G[s][j][k]
                               for s in range(2)))
    R1212 = sp.expand(sum(g[0, s] * Riem(s, 1, 0, 1) for s in range(2)))
    return sp.simplify(trunc(R1212) / sp.simplify(g.det()))


def lift_curved(Acomp, c, g, G):
    """Lam_g(A) in the canonical Darboux frame (e_1,e_2) of PT^*R^2.

    The metric frame of D is (e_1^g, e_2^g) = (d_theta, the horizontal lift of the
    g-unit vector spanning the line); in it the components of the lift are
    (c kappa_A, 0, kappa_A) -- that is the definition of Subsec. 3.8.  Both frames
    frame the same D, so the conversion is a 2x2 matrix.
    """
    v = sp.Matrix([1, m])                                   # spans the line
    gv = trunc(sp.expand((v.T * g * v)[0, 0]))              # g(v,v)
    A = sp.Matrix([[Acomp[0], Acomp[1]], [Acomp[1], Acomp[2]]])
    kap = sp.together(sp.expand((v.T * A * v)[0, 0]) / gv)  # kappa_A = A(v,v)/g(v,v)

    # e_1^g = d_theta.  theta is the g-angle of the line; with u(theta) a g-unit
    # vector turning at unit g-speed, the slope is m = u_2/u_1 and
    #   d m / d theta = (u_1 u_2' - u_2 u_1') / u_1^2 = J/u_1^2,
    # where J is the area (= g-rotation) factor: for the unit vector, u' is the
    # g-rotation of u, i.e. u' = (sqrt(det g)/1) * g^{-1} (-u_2, u_1)^T.
    detg = sp.sqrt(sp.simplify(g.det()))
    u = v / sp.sqrt(gv)
    rot = sp.simplify(detg) * g.inv() * sp.Matrix([-u[1], u[0]])   # the g-rotation
    dm_dth = trunc(sp.expand(sp.simplify((u[0] * rot[1] - u[1] * rot[0]) / u[0] ** 2)))
    E1g = (sp.S(0), sp.S(0), dm_dth)                        # (d_x, d_y, d_m)

    # e_2^g = the Levi-Civita horizontal lift of the unit vector u.
    # Moving along u, a parallel v changes by (Dv)^i = -Gamma^i_{jk} u^j v^k, and the
    # slope coordinate by  m-dot = (Dv)^2 - m (Dv)^1  (v = (1,m) up to scale).
    Dv = [trunc(-sum(G[i][j][k] * u[j] * v[k] for j in range(2) for k in range(2)))
          for i in range(2)]
    mdot = trunc(sp.expand(Dv[1] - m * Dv[0]))
    E2g = (u[0], u[1], mdot)

    # both frames frame D: e_i^g = M_{i1} e_1 + M_{i2} e_2 with e_1 = d_m,
    # e_2 = d_x + m d_y.  Reading the (d_x, d_y, d_m) components:
    #     e_2-coefficient = the d_x one;  e_1-coefficient = d_m minus m * (d_x) * 0.
    # (the Reeb component must vanish, which is the check that e_i^g lie in D.)
    Mrows = []
    for V in (E1g, E2g):
        reeb = trunc(sp.expand(V[1] - m * V[0]))
        Mrows.append((V[2], V[0], reeb))
    Mat = sp.Matrix([[Mrows[0][0], Mrows[0][1]], [Mrows[1][0], Mrows[1][1]]])
    Qg = sp.Matrix([[c * kap, 0], [0, kap]])
    Minv = Mat.inv()
    Qe = sp.simplify(Minv * Qg * Minv.T)
    comps = {(1, 1): Qe[0, 0], (1, 2): Qe[0, 1], (2, 2): Qe[1, 1]}
    return comps, kap, [r[2] for r in Mrows]


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prop39curv.log"), "w")
    P.LOG = LOG

    say("=== 0. the operator, from the solve of prop39.py")
    op0, op2 = P.part2()
    terms, bas = op2[0], [op2[1]]
    P.verify_invariance(op2, 0, 0, 3, "kGz")
    CHECKS[0] += P.CHECKS[0]
    CHECKS[1] += P.CHECKS[1]

    say("\n=== 1. the metric in normal coordinates")
    g = metric()
    gi, G = christoffel(g)
    say("  g = %s" % str(g.tolist()))
    check("the Gauss curvature at the origin is K",
          sp.simplify(gauss(g, G).subs({x: 0, y: 0}) - K) == 0)
    check("the Christoffels vanish at the origin (normal coordinates)",
          all(sp.simplify(G[i][j][k].subs({x: 0, y: 0})) == 0
              for i in range(2) for j in range(2) for k in range(2)))
    check("their first derivatives do not (the curvature is there)",
          any(sp.simplify(sp.diff(G[i][j][k], x).subs({x: 0, y: 0})) != 0
              for i in range(2) for j in range(2) for k in range(2)))

    say("\n=== 2. the lift for a curved metric, in the canonical Darboux frame")
    c = sp.Symbol("c")
    A_, B_ = jets(7), jets(8)
    qA, kA, reebA = lift_curved(A_, c, g, G)
    qB, kB, _ = lift_curved(B_, c, g, G)
    check("the metric frame lies in the contact distribution (the Reeb components "
          "vanish)", all(sp.simplify(r) == 0 for r in reebA))
    flatA, kA0 = P.lift(A_, c)
    check("at K = 0 the curved lift is the flat one",
          all(sp.simplify(sp.together(qA[k].subs(K, 0) - flatA[k])) == 0 for k in KEYS))

    say("\n=== 3. kGz(Lam_g A, Lam_g B) with the curvature in")
    # kGz is bilinear, so split the ARGUMENTS in K rather than the answer:
    #   Lam = Lam^(0) + K Lam^(1) + O(K^2)  =>  the K-part of the answer is
    #   kGz(Lam^(1)A, Lam^(0)B) + kGz(Lam^(0)A, Lam^(1)B).
    split = lambda q: ({k: sp.cancel(sp.together(q[k].subs(K, 0))) for k in KEYS},
                       {k: sp.cancel(sp.together(sp.diff(q[k], K).subs(K, 0)))
                        for k in KEYS})
    qA0, qA1 = split(qA)
    qB0, qB1 = split(qB)
    flatB = P.lift(B_, c)[0]
    check("the K = 0 part of the curved lift of B is the flat one",
          all(sp.simplify(sp.together(qB0[k] - flatB[k])) == 0 for k in KEYS))
    at0 = lambda e: sp.cancel(sp.together(sp.together(e).subs({x: 0, y: 0})))
    val0 = build_op(terms, bas[0], qA0, qB0)
    dK = sp.simplify(at0(build_op(terms, bas[0], qA1, qB0)
                         + build_op(terms, bas[0], qA0, qB1)))
    val = val0 + K * (build_op(terms, bas[0], qA1, qB0)
                      + build_op(terms, bas[0], qA0, qB1))
    say("  the curvature part, d/dK at the origin:")
    say("      %s" % sp.factor(sp.simplify(dK)))
    check("the lift itself DOES depend on the curvature (so the test is not vacuous)",
          any(sp.simplify(qA1[k]) != 0 for k in KEYS))
    say("  the two halves of the K-term separately:")
    h1 = sp.simplify(at0(build_op(terms, bas[0], qA1, qB0)))
    h2 = sp.simplify(at0(build_op(terms, bas[0], qA0, qB1)))
    say("      kGz(Lam^(1)A, Lam^(0)B) = %s" % sp.factor(h1))
    say("      kGz(Lam^(0)A, Lam^(1)B) = %s" % sp.factor(h2))
    check("each half vanishes SEPARATELY -- stronger than a cancellation",
          sp.simplify(h1) == 0 and sp.simplify(h2) == 0)
    say("  why: the curvature part of the lift, in the Darboux frame, is")
    for k in KEYS:
        say("      q^(1)_%d%d = %s" % (k[0], k[1], sp.factor(sp.simplify(qA1[k]))))
    check("the horizontal component does not feel the curvature at all: it is the "
          "pull-back A(v,v), into which no metric enters",
          sp.simplify(qA1[(2, 2)]) == 0)
    check("the vertical and the mixed components do feel it (and their derivatives "
          "at the origin are non-zero, so the operator sees them)",
          sp.simplify(qA1[(1, 1)]) != 0 and sp.simplify(qA1[(1, 2)]) != 0
          and sp.simplify(sp.diff(qA1[(1, 2)], y).subs({x: 0, y: 0})) != 0)
    check("and they cancel: the curvature term of the answer is ZERO",
          sp.simplify(dK) == 0)
    say("  the same on three further independent pairs of 2-jets:")
    for s_ in (11, 12, 13):
        A2, B2 = jets(1000 + s_), jets(1100 + s_)
        qa, _, _ = lift_curved(A2, c, g, G)
        qb, _, _ = lift_curved(B2, c, g, G)
        qa0, qa1 = split(qa)
        qb0, qb1 = split(qb)
        d2 = sp.simplify(at0(build_op(terms, bas[0], qa1, qb0)
                             + build_op(terms, bas[0], qa0, qb1)))
        check("no curvature term for the jet pair %d" % s_, sp.simplify(d2) == 0)
    check("at K = 0 it is the flat answer, the one prop39.py computed",
          sp.simplify(at0(val0) - at0(build_op(terms, bas[0], flatA, flatB))) == 0)

    say("\n=== 4. what a curvature term could have been, and is not")
    # A curvature term is K times a pointwise bilinear expression in (A,B); the
    # antisymmetric ones are spanned by  tr A kappa_B - kappa_A tr B  (and kappa
    # kappa is symmetric, so it cannot occur).
    trA = sp.expand(A_[0] + A_[2])
    trB = sp.expand(B_[0] + B_[2])
    cand = {
        "tr A kappa_B - kappa_A tr B": at0(trA * kB.subs(K, 0) - kA.subs(K, 0) * trB),
        "kappa_A kappa_B (symmetric, should not occur)":
            at0(kA.subs(K, 0) * kB.subs(K, 0)),
        "{kappa_A, kappa_B}": at0(bracket(kA.subs(K, 0), kB.subs(K, 0))),
    }
    for nm, ex in cand.items():
        say("      %-46s : %s" % (nm, sp.simplify(sp.factor(ex))))
    ps = sp.symbols("r0:%d" % len(cand))
    eqs = sp.Poly(sp.expand(sp.numer(sp.cancel(sp.together(
        dK - sum(p * e for p, e in zip(ps, cand.values())))))), m).coeffs()
    sol = sp.solve(eqs, ps, dict=True)
    say("  fit of the (vanishing) curvature term in those candidates: %s" % sol)
    check("the fit returns zero coefficients, as it must",
          bool(sol) and all(sp.simplify(sol[0].get(p, p)) == 0 for p in ps))
    say("  So the answer for a curved metric is the flat expression, verbatim:")
    say("      kGz(Lam_g A, Lam_g B) = 2c [ (d_theta kA)(nabla kB) - (nabla kA)"
        "(d_theta kB) - (R_theta kA) kB + kA (R_theta kB) ]")

    say("\n=== 5. harmonics of the whole answer")
    hs = harmonics(at0(sp.together(val.subs(K, sp.Rational(2, 5))
                                   .subs(c, sp.Rational(3, 2)))), 3)
    say("      harmonics of kGz(Lam A, Lam B) with K = 2/5 : %s" % hs)
    check("no harmonic above 6 with the curvature in", hs and max(hs) <= 6)

    say("\n%d/%d checks pass" % (CHECKS[1], CHECKS[0]))
    LOG.close()


if __name__ == "__main__":
    main()
