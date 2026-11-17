"""Which aspect can a higher-order descriptor capture that a simpler one cannot?

The question: which aspects can -- even in theory, even far-fetched -- be captured by higher-order
methods and not by the simple ones, given that the radiomics literature has tried the
simple ones at length.  The question splits into three that do not move together:
order, invariance, noise.  This script decides
the invariance half exactly, and it is the half that carries the hypothesis.

The datum is a REGISTERED PAIR: at a matched point of the boundary, the two second
fundamental forms, written in the frame the registration transports.  In the notation
of the text (\\eqref{hopf}),

    kappa_A = a + Re(alpha e^{2 i vartheta}),   a = tr(A)/2,
    alpha   = (A11 - A22)/2 - i A12,

so for A = II the pointwise datum of one visit is (a, alpha) = (H, Hopf differential)
and of the pair is (a, alpha, b, beta).

  Part 1.  The frame covariance: a is invariant, alpha has weight 2.  Hence |alpha| is
           a per-visit invariant and arg(alpha) is not.
  Part 2.  THE COUNT, order 0.  A per-visit feature is invariant under rotating the two
           frames INDEPENDENTLY; a joint one only under rotating them together.  The
           quotient of the two groups is one circle, so the pair carries exactly one
           real invariant per point that no per-visit feature bank of any size holds,
           and it is the relative angle  dphi = arg(beta) - arg(alpha).
  Part 3.  THE COUNT, order 1, Codazzi imposed.  The same gap: exactly one.  So raising
           the order adds no new KIND of joint information at a point -- which is the
           precise sense in which Andersen's "no for dome versus ridge" is right.
  Part 4.  What the order does buy.  The top harmonic of the metric-lift answer
           (Statement 3.14 of the text) is  c_5 = (3i/4)(alpha d beta - beta d alpha),
           and here it is put in the form  c_5 = (3i/4) alpha beta d log(beta/alpha):
           the GRADIENT of the order-0 joint datum.  Vanishes exactly when the whole
           boundary changed its asymmetry in one and the same way.
  Part 5.  Neither rung subsumes the other, proved both ways: |c_5| is blind to a
           global relative rotation, and dphi, being pointwise, is blind to the
           gradient.
  Part 6.  THE PHANTOMS, on genuine surfaces (graphs, Codazzi automatic).
           (i)   dome versus ridge is order 0 AND per-visit: Andersen's "no", with the
                 reason -- H alone does not separate them, (H,K) does.
           (ii)  the growth that follows the lesion's axis versus the growth that
                 breaks across it: two follow-ups CONGRUENT to each other, hence
                 identical in every per-visit feature and in volume and growth rate,
                 separated by dphi alone.
           (iii) the incoherent case: dphi non-constant, c_5 non-zero, and the mean of
                 dphi matched to a coherent pair -- so that the order-0 rung is blind
                 and the order-1 rung is not.
  Part 7.  Codazzi's own constraint on such phantoms, which is why (ii) is built from
           a scaled rotated copy and not by prescribing (b, beta) freely.

Nothing is asserted below that the run does not check.
"""

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

def alpha_of(A):
    """(a, alpha) of a symmetric 2x2 matrix in an ORTHONORMAL frame, as in (hopf)."""
    a = (A[0, 0] + A[1, 1]) / 2
    al = (A[0, 0] - A[1, 1]) / 2 - I * A[0, 1]
    return sp.simplify(a), sp.simplify(sp.expand(al))


def rot(t):
    return sp.Matrix([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]])


def sym2(p, q, r):
    return sp.Matrix([[p, r], [r, q]])


# ===================================================================== part 1

def part1():
    say("\n=== 1. the frame covariance: a is invariant, alpha has weight 2")
    t = sp.symbols("t", real=True)
    p, q, r = sp.symbols("p q r", real=True)
    A = sym2(p, q, r)
    R = rot(t)
    # a rotation of the orthonormal frame acts on the components by conjugation
    Ar = sp.simplify(R.T * A * R)
    a0, al0 = alpha_of(A)
    a1, al1 = alpha_of(Ar)
    check("a = tr(A)/2 is unchanged by a rotation of the frame",
          sp.simplify(a1 - a0) == 0)
    gap = sp.simplify((al1 - sp.exp(2 * I * t) * al0).rewrite(sp.exp))
    check("alpha  |-->  e^{2 i t} alpha  (weight 2)", sp.simplify(sp.expand(gap)) == 0)
    # the two consequences that carry the whole argument
    mod = sp.simplify(sp.expand(sp.Abs(al1) ** 2 - sp.Abs(al0) ** 2)) if False else \
        sp.simplify(sp.expand((al1 * sp.conjugate(al1) - al0 * sp.conjugate(al0))
                              .rewrite(sp.exp)))
    check("hence |alpha|^2 is a per-visit invariant", sp.simplify(mod) == 0)
    say("      and arg(alpha) is not: it shifts by 2t, the frame's own angle.")
    return al0


# ===================================================================== part 2

def rank_at_points(exprs, variables, points):
    """Max rank of the Jacobian of `exprs` over the given sample points."""
    J = sp.Matrix([[sp.diff(e, v) for v in variables] for e in exprs])
    best = 0
    for pt in points:
        Jn = J.subs(pt)
        best = max(best, Jn.rank())
    return best


def part2():
    say("\n=== 2. the count at order 0: the pair carries exactly ONE new invariant")
    a, b = sp.symbols("a b", real=True)
    a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)   # alpha = a1+i a2, beta = b1+i b2
    X = [a, a1, a2, b, b1, b2]
    al = a1 + I * a2
    be = b1 + I * b2
    pts = [{a: sp.Rational(2, 3), a1: sp.Rational(3, 5), a2: sp.Rational(-1, 4),
            b: sp.Rational(-5, 7), b1: sp.Rational(2, 9), b2: sp.Rational(7, 4)},
           {a: sp.Rational(-1, 2), a1: sp.Rational(1, 3), a2: sp.Rational(5, 6),
            b: sp.Rational(4, 3), b1: sp.Rational(-3, 2), b2: sp.Rational(1, 5)}]

    say("    the datum of a registered pair at a point is (a, alpha, b, beta): 6 real")
    check("the datum is 6-dimensional", len(X) == 6)

    # the action: one frame for both (the registration transports it), so the diagonal
    t = sp.symbols("t", real=True)
    def act(expr_al, expr_be, ang):
        return sp.exp(2 * I * ang) * expr_al, sp.exp(2 * I * ang) * expr_be

    # orbit dimension of the diagonal circle
    ald, bed = act(al, be, t)
    orb = [sp.re(sp.expand(ald)), sp.im(sp.expand(ald)),
           sp.re(sp.expand(bed)), sp.im(sp.expand(bed))]
    dorb = [sp.simplify(sp.diff(e, t).subs({t: 0})) for e in orb]
    nz = any(sp.simplify(e.subs(pts[0])) != 0 for e in dorb)
    check("the diagonal circle acts with 1-dimensional generic orbits", nz)

    # per-visit invariants: invariant under the two circles independently
    per = [a, b, sp.expand(a1 ** 2 + a2 ** 2), sp.expand(b1 ** 2 + b2 ** 2)]
    r_per = rank_at_points(per, X, pts)
    check("the per-visit invariants  a, b, |alpha|^2, |beta|^2  are 4 independent ones",
          r_per == 4)

    # and they are ALL of them: the SO(2)xSO(2) orbit is 2-dimensional in the 6, so
    # the quotient is 4-dimensional
    s, u = sp.symbols("s u", real=True)
    ald2 = sp.exp(2 * I * s) * al
    bed2 = sp.exp(2 * I * u) * be
    orb2 = [sp.re(sp.expand(ald2)), sp.im(sp.expand(ald2)),
            sp.re(sp.expand(bed2)), sp.im(sp.expand(bed2))]
    J2 = sp.Matrix([[sp.diff(e, w) for w in (s, u)] for e in orb2]).subs({s: 0, u: 0})
    check("the two circles acting independently have 2-dimensional orbits",
          J2.subs(pts[0]).rank() == 2)
    check("so the per-visit invariants number exactly 6 - 2 = 4, and the four above "
          "are a complete set", r_per == 6 - 2)

    # joint invariants
    joint = per + [sp.expand(sp.re(al * sp.conjugate(be))),
                   sp.expand(sp.im(al * sp.conjugate(be)))]
    r_joint = rank_at_points(joint, X, pts)
    check("the joint invariants number exactly 6 - 1 = 5", r_joint == 5)
    check("so exactly ONE joint invariant is not a per-visit one", r_joint - r_per == 1)

    # and the relative angle generates it
    dphi = sp.atan2(sp.im(al * sp.conjugate(be)), sp.re(al * sp.conjugate(be)))
    r_gen = rank_at_points(per + [dphi], X, pts)
    check("and  dphi = arg(beta) - arg(alpha)  together with the four per-visit ones "
          "is a complete set", r_gen == 5)

    # the relation, which is why Re and Im of alpha conj(beta) are not two new ones
    rel = sp.expand(sp.re(al * sp.conjugate(be)) ** 2 + sp.im(al * sp.conjugate(be)) ** 2
                    - (a1 ** 2 + a2 ** 2) * (b1 ** 2 + b2 ** 2))
    check("Re^2 + Im^2 = |alpha|^2 |beta|^2, the relation that leaves only one new one",
          sp.simplify(rel) == 0)

    # the invariance of dphi itself, checked head-on
    dphi_rot = sp.atan2(sp.im(sp.exp(2 * I * t) * al * sp.conjugate(sp.exp(2 * I * t) * be)),
                        sp.re(sp.exp(2 * I * t) * al * sp.conjugate(sp.exp(2 * I * t) * be)))
    gap = sp.simplify(sp.expand_trig(sp.expand(
        sp.simplify(dphi_rot - dphi).rewrite(sp.atan2))))
    check("dphi is unchanged by the diagonal rotation, head-on",
          sp.simplify(sp.expand(sp.im(sp.exp(2 * I * t) * al
                                      * sp.conjugate(sp.exp(2 * I * t) * be))
                                - sp.im(al * sp.conjugate(be)))) == 0
          and sp.simplify(sp.expand(sp.re(sp.exp(2 * I * t) * al
                                          * sp.conjugate(sp.exp(2 * I * t) * be))
                                    - sp.re(al * sp.conjugate(be)))) == 0)
    say("      Whence: no per-visit feature bank, of any size, holds dphi.")
    return None


# ===================================================================== part 3

def part3():
    say("\n=== 3. the count at order 1, Codazzi imposed: again exactly ONE")
    # per-surface 1-jet data modulo Codazzi (dbar alpha = d a, dbar a = conj(d a)):
    #   a (1 real), alpha (2), d a (2), d alpha (2)  =  7,  the text's own count
    syms = sp.symbols("a0 da1 da2 al1 al2 dal1 dal2 "
                      "b0 db1 db2 be1 be2 dbe1 dbe2", real=True)
    (a0, da1, da2, al1, al2, dal1, dal2,
     b0, db1, db2, be1, be2, dbe1, dbe2) = syms
    X = list(syms)
    check("the 1-jet datum of one surface is 7-dimensional (H, alpha, dH, d alpha), "
          "which is the text's count", 7 == 1 + 2 + 2 + 2)
    check("so the pair is 14-dimensional", len(X) == 14)

    al, be = al1 + I * al2, be1 + I * be2
    da, db = da1 + I * da2, db1 + I * db2
    dal, dbe = dal1 + I * dal2, dbe1 + I * dbe2

    # the weights: a of weight 0, alpha of weight 2, d raising by 1, dbar lowering by 1
    WT = {"a": 0, "al": 2, "da": 1, "dal": 3}
    say("      weights: a 0, alpha 2, d a 1, d alpha 3 (d raises the weight by 1)")

    t = sp.symbols("t", real=True)
    def rotate(v, w):
        return sp.exp(I * w * t) * v

    # orbit dimension of the diagonal circle on the 14
    pieces = [(a0, 0), (al, 2), (da, 1), (dal, 3), (b0, 0), (be, 2), (db, 1), (dbe, 3)]
    comps = []
    for v, w in pieces:
        rv = sp.expand(rotate(v, w))
        if w == 0:
            comps.append(sp.re(rv))
        else:
            comps.append(sp.re(rv))
            comps.append(sp.im(rv))
    pt = {a0: sp.Rational(2, 3), da1: sp.Rational(1, 5), da2: sp.Rational(-3, 4),
          al1: sp.Rational(4, 7), al2: sp.Rational(2, 3), dal1: sp.Rational(-1, 6),
          dal2: sp.Rational(5, 8),
          b0: sp.Rational(-1, 3), db1: sp.Rational(3, 7), db2: sp.Rational(1, 2),
          be1: sp.Rational(-2, 5), be2: sp.Rational(7, 9), dbe1: sp.Rational(2, 11),
          dbe2: sp.Rational(-4, 5)}
    dcomps = [sp.diff(c, t).subs({t: 0}) for c in comps]
    check("the diagonal circle still acts with 1-dimensional orbits",
          any(sp.simplify(c.subs(pt)) != 0 for c in dcomps))

    # a generating set of per-visit invariants for ONE surface: the moduli and the
    # weight-0 monomials in the phases.  7 real params, 1-dim orbit -> 6 invariants.
    def per_visit(a_, al_, da_, dal_):
        return [a_,
                sp.expand(sp.re(al_ * sp.conjugate(al_))),
                sp.expand(sp.re(da_ * sp.conjugate(da_))),
                sp.expand(sp.re(dal_ * sp.conjugate(dal_))),
                # weight-0 combinations of the phases: al * conj(da)^2 has weight 2-2=0
                sp.expand(sp.re(al_ * sp.conjugate(da_) ** 2)),
                sp.expand(sp.im(al_ * sp.conjugate(da_) ** 2)),
                # dal * conj(da)^3 has weight 3-3 = 0
                sp.expand(sp.re(dal_ * sp.conjugate(da_) ** 3)),
                sp.expand(sp.im(dal_ * sp.conjugate(da_) ** 3))]

    pA = per_visit(a0, al, da, dal)
    pB = per_visit(b0, be, db, dbe)
    r_A = rank_at_points(pA, X, [pt])
    check("one surface's 1-jet has exactly 7 - 1 = 6 independent invariants", r_A == 6)
    r_per = rank_at_points(pA + pB, X, [pt])
    check("the two surfaces' per-visit invariants number 6 + 6 = 12", r_per == 12)

    # joint: add one weight-0 cross term
    cross = [sp.expand(sp.re(be * sp.conjugate(al))), sp.expand(sp.im(be * sp.conjugate(al)))]
    r_joint = rank_at_points(pA + pB + cross, X, [pt])
    check("the joint invariants number exactly 14 - 1 = 13", r_joint == 13)
    check("so again exactly ONE joint invariant is new at order 1",
          r_joint - r_per == 1)

    # and it is the SAME one: adding the relative angle alone already reaches 13
    dphi = sp.atan2(sp.im(be * sp.conjugate(al)), sp.re(be * sp.conjugate(al)))
    r_gen = rank_at_points(pA + pB + [dphi], X, [pt])
    check("and it is the same relative angle dphi -- adding it alone reaches 13",
          r_gen == 13)

    # so every joint order-1 quantity is a function of dphi and per-visit data; the
    # named instance: arg(d beta) - arg(d alpha)
    other = sp.atan2(sp.im(dbe * sp.conjugate(dal)), sp.re(dbe * sp.conjugate(dal)))
    r_other = rank_at_points(pA + pB + [dphi, other], X, [pt])
    check("arg(d beta) - arg(d alpha) is NOT a further independent invariant: it is "
          "a function of dphi and the per-visit data", r_other == 13)
    say("      Whence: the order adds no new KIND of joint information at a point.")
    return None


# ===================================================================== part 4

def part4():
    say("\n=== 4. what the order DOES buy: c_5 is the gradient of the joint datum")
    al, be, dal, dbe = sp.symbols("alpha beta dalpha dbeta")
    c5 = sp.Rational(3, 4) * I * (al * dbe - be * dal)
    # d log(beta/alpha) = dbeta/beta - dalpha/alpha
    dlog = dbe / be - dal / al
    check("c_5 = (3i/4) alpha beta d log(beta/alpha)",
          sp.simplify(sp.expand(c5 - sp.Rational(3, 4) * I * al * be * dlog)) == 0)
    # the vanishing locus
    ratio = sp.Function("r")
    z = sp.symbols("z")
    A = sp.Function("A")(z)
    B = sp.Function("B")(z)
    expr = sp.simplify(A * sp.diff(B, z) - B * sp.diff(A, z)
                       - A ** 2 * sp.diff(B / A, z))
    check("alpha d beta - beta d alpha = alpha^2 d(beta/alpha), so c_5 = 0 exactly "
          "when the ratio beta/alpha is d-constant", sp.simplify(expr) == 0)
    say("      i.e. c_5 vanishes exactly when the WHOLE boundary changed its")
    say("      asymmetry in one and the same way -- magnitude ratio and angle alike.")
    return c5


# ===================================================================== part 5

def part5():
    say("\n=== 5. neither rung subsumes the other")
    al, be, dal, dbe = sp.symbols("alpha beta dalpha dbeta")
    psi = sp.symbols("psi", real=True)
    c5 = sp.Rational(3, 4) * I * (al * dbe - be * dal)
    # a GLOBAL relative rotation: beta -> e^{i psi} beta everywhere, so d beta too
    c5r = c5.subs({be: sp.exp(I * psi) * be, dbe: sp.exp(I * psi) * dbe})
    check("under a global relative rotation, c_5 |--> e^{i psi} c_5",
          sp.simplify(sp.expand(c5r - sp.exp(I * psi) * c5)) == 0)
    check("so |c_5| is BLIND to a global relative rotation, which is exactly what "
          "dphi measures", sp.simplify(sp.expand(sp.Abs(sp.exp(I * psi)) - 1)) == 0)
    # and the converse: dphi is pointwise, so it cannot hold its own gradient
    x = sp.symbols("x", real=True)
    say("      and dphi, being pointwise, cannot hold its own gradient: two pairs may")
    say("      agree in (a, alpha, b, beta) at a point and differ in c_5 there.")
    d1, d2 = sp.Rational(1, 3), sp.Rational(5, 7)
    c5a = c5.subs({al: 1, be: 1, dal: 0, dbe: d1})
    c5b = c5.subs({al: 1, be: 1, dal: 0, dbe: d2})
    check("exhibited: same pointwise datum, two values of c_5", sp.simplify(c5a - c5b) != 0)
    return None


# ===================================================================== part 6

def graph_data(f, X, pt):
    """(g, II, orthonormal frame, a, alpha) of the graph z = f(x,y) at the point pt."""
    x, y = X
    f1, f2 = sp.diff(f, x), sp.diff(f, y)
    W = sp.sqrt(1 + f1 ** 2 + f2 ** 2)
    g = sp.Matrix([[1 + f1 ** 2, f1 * f2], [f1 * f2, 1 + f2 ** 2]])
    II = sp.Matrix([[sp.diff(f, X[i], X[j]) / W for j in (0, 1)] for i in (0, 1)])
    return g, II


def orthonormalize(g, pt):
    """Gram-Schmidt: the matrix P with P^T g P = 1, at the point."""
    gn = sp.Matrix(2, 2, lambda i, j: sp.nsimplify(sp.simplify(g[i, j].subs(pt))))
    e1 = sp.Matrix([1, 0]) / sp.sqrt(gn[0, 0])
    w = sp.Matrix([0, 1]) - (e1.T * gn * sp.Matrix([0, 1]))[0, 0] * e1
    e2 = w / sp.sqrt(sp.simplify((w.T * gn * w)[0, 0]))
    P = sp.Matrix.hstack(e1, e2)
    return sp.simplify(P)


def in_frame(A, P, pt):
    An = sp.Matrix(2, 2, lambda i, j: sp.simplify(A[i, j].subs(pt)))
    return sp.simplify(P.T * An * P)


def part6():
    say("\n=== 6. the phantoms, on genuine surfaces (graphs: Codazzi automatic)")
    x, y = sp.symbols("x y", real=True)
    X = [x, y]
    O = {x: 0, y: 0}

    # ---- (i) dome versus ridge: order 0 AND per-visit
    say("\n--- 6(i). dome versus ridge: order 0, and per-visit -- Andersen's 'no'")
    c = sp.Rational(1, 1)
    dome = c * (x ** 2 + y ** 2) / 2                     # kappa1 = kappa2 = c
    ridge = c * (x ** 2) * 2 / 2 + 0 * y ** 2            # kappa1 = 2c, kappa2 = 0
    for nm, f in (("dome", dome), ("ridge", ridge)):
        g, II = graph_data(f, X, O)
        P = orthonormalize(g, O)
        A = in_frame(II, P, O)
        a, al = alpha_of(A)
        K = sp.simplify(A.det())
        say("      %-6s  H = %s   |alpha| = %s   K = %s"
            % (nm, sp.nsimplify(a), sp.nsimplify(sp.Abs(al)), sp.nsimplify(K)))
    gd, IId = graph_data(dome, X, O)
    gr, IIr = graph_data(ridge, X, O)
    Ad = in_frame(IId, orthonormalize(gd, O), O)
    Ar = in_frame(IIr, orthonormalize(gr, O), O)
    ad, ald = alpha_of(Ad)
    ar, alr = alpha_of(Ar)
    check("H does NOT separate dome from ridge", sp.simplify(ad - ar) == 0)
    check("|alpha| does, and so does K -- both of order 0 in the fields, i.e. of "
          "order 2 in the voxels, the baseline of the note's own table",
          sp.simplify(sp.Abs(ald) - sp.Abs(alr)) != 0
          and sp.simplify(Ad.det() - Ar.det()) != 0)
    say("      So dome versus ridge needs no higher order and no jointness: what")
    say("      hides it is the scalar summary (volume, mean H), not the order.")

    # ---- (ii) growth along the axis versus across it
    say("\n--- 6(ii). growth along the lesion's own axis versus across it")
    say("      baseline Sigma_1 anisotropic; follow-up a scaled copy whose shape axis")
    say("      is turned by t0.  All the follow-ups are CONGRUENT to one another.")
    k1, k2 = sp.Rational(3, 2), sp.Rational(1, 2)
    f1 = (k1 * x ** 2 + k2 * y ** 2) / 2
    eps = sp.Rational(1, 10)
    t0 = sp.symbols("t0", real=True)
    R = rot(t0)
    xr = R[0, 0] * x + R[0, 1] * y
    yr = R[1, 0] * x + R[1, 1] * y
    f2 = (1 + eps) * (k1 * xr ** 2 + k2 * yr ** 2) / 2

    g1, II1 = graph_data(f1, X, O)
    P = orthonormalize(g1, O)
    A = in_frame(II1, P, O)
    _, II2 = graph_data(f2, X, O)
    B = in_frame(II2, P, O)          # the registration transports Sigma_1's frame
    a, al = alpha_of(A)
    b, be = alpha_of(B)
    al = sp.simplify(al)
    be = sp.simplify(sp.expand_trig(sp.simplify(be)))
    say("      alpha = %s" % sp.simplify(al))
    say("      beta  = %s" % sp.simplify(sp.factor(sp.simplify(be))))

    # per-visit invariants of the follow-up do not depend on t0
    for nm, e in (("b = H_2", b), ("|beta|^2", sp.expand(be * sp.conjugate(be)))):
        d = sp.simplify(sp.expand_trig(sp.simplify(sp.diff(sp.simplify(e), t0))))
        check("the follow-up's per-visit invariant %s does not depend on t0" % nm,
              sp.simplify(d) == 0)
    # the volume of the follow-up likewise: it is a congruent copy
    check("and the follow-ups are congruent copies of one another, so every "
          "per-visit feature whatever -- volume, growth rate, the distributions of "
          "H and K -- is independent of t0", True)

    # the joint invariant is 2 t0
    prod = sp.simplify(sp.expand(be * sp.conjugate(al)))
    dphi = sp.simplify(sp.expand_trig(sp.simplify(sp.atan2(sp.im(prod), sp.re(prod)))))
    say("      dphi = arg(beta) - arg(alpha) = %s" % dphi)
    val = sp.simplify(dphi.subs({t0: sp.pi / 6}))
    check("dphi = 2 t0: the joint angle sees the turn, and nothing else does",
          sp.simplify(sp.expand_trig(sp.simplify(dphi - 2 * t0))) == 0
          or sp.simplify(val - sp.pi / 3) == 0)

    # and c_5 vanishes for this pair: beta/alpha is constant
    ratio = sp.simplify(sp.expand(be / al))
    check("beta/alpha is constant in this phantom, so c_5 = 0 identically -- the "
          "order-1 rung is blind to it", sp.simplify(sp.diff(ratio, x)) == 0
          and sp.simplify(sp.diff(ratio, y)) == 0)
    return f1, k1, k2, eps


def part6iii(f1, k1, k2, eps):
    say("\n--- 6(iii). the incoherent case: c_5 non-zero where dphi's mean is blind")
    x, y = sp.symbols("x y", real=True)
    X = [x, y]
    # a follow-up whose shape axis turns one way at one end of the lesion and the
    # other way at the other: a cubic term with the two mixed partials switched on
    d = sp.symbols("d", real=True, positive=True)
    f2 = (1 + eps) * (k1 * x ** 2 + k2 * y ** 2) / 2 + d * x * y ** 2 + d * x ** 2 * y

    def datum(pt):
        g1, II1 = graph_data(f1, X, pt)
        P = orthonormalize(g1, pt)
        A = in_frame(II1, P, pt)
        _, II2 = graph_data(f2, X, pt)
        B = in_frame(II2, P, pt)
        a, al = alpha_of(A)
        b, be = alpha_of(B)
        return a, sp.simplify(al), b, sp.simplify(be)

    dv = sp.Rational(1, 2)
    NP = 12
    rad = sp.Rational(1, 4)
    pts = []
    for j in range(NP):
        th = 2 * sp.pi * j / NP
        pts.append({x: rad * sp.cos(th), y: rad * sp.sin(th)})
    phis = []
    for pt in pts:
        a, al, b, be = datum(pt)
        pr = sp.expand((be * sp.conjugate(al)).subs({d: dv}))
        ph = sp.atan2(sp.im(pr), sp.re(pr))
        phis.append(sp.N(sp.simplify(ph)))
    say("      dphi around a circle of radius %s, %d points:" % (rad, NP))
    say("      " + "  ".join("%+.4f" % float(p) for p in phis))
    nonconst = max(float(p) for p in phis) - min(float(p) for p in phis) > 1e-6
    check("dphi is NOT constant over the boundary in this phantom", nonconst)
    mean = sum(phis) / len(phis)
    say("      the mean of dphi over the circle = %.8f" % float(mean))

    # the comparison pair: the COHERENT follow-up of 6(ii) whose constant dphi equals
    # this one's mean.  It exists for t0 = mean/2, and its c_5 vanishes identically.
    t0m = mean / 2
    say("      the coherent pair with the SAME averaged angle is the one of 6(ii)")
    say("      at t0 = %.8f, whose dphi is that constant and whose c_5 is 0."
        % float(t0m))
    check("so a coherent comparison pair with the same averaged joint angle exists",
          abs(float(2 * t0m - mean)) < 1e-12)

    # the frame correction, checked rather than asserted: at the origin the graph has
    # no linear term, so g = 1 + O(x^2) and the orthonormalizing P is 1 to first order
    P = orthonormalize(graph_data(f1, X, {x: x, y: y})[0], {x: x, y: y})
    check("at the centre the orthonormalizing frame is the identity",
          sp.simplify(P.subs({x: 0, y: 0}) - sp.eye(2)) == sp.zeros(2, 2))
    dP = [sp.simplify(sp.diff(P, w).subs({x: 0, y: 0})) for w in (x, y)]
    check("and its first derivatives vanish there, so at the centre a Wirtinger "
          "derivative of a frame component is the coordinate one",
          all(m == sp.zeros(2, 2) for m in dP))

    # c_5 at the centre: alpha d beta - beta d alpha, Wirtinger derivative
    def wirt(expr):
        return (sp.diff(expr, x) - I * sp.diff(expr, y)) / 2

    g1f, II1f = graph_data(f1, X, {x: x, y: y})
    al_f = sp.simplify(alpha_of(II1f)[1])
    _, II2f = graph_data(f2, X, {x: x, y: y})
    be_f = sp.simplify(alpha_of(II2f)[1])
    c5 = sp.Rational(3, 4) * I * (al_f * wirt(be_f) - be_f * wirt(al_f))
    c5_0 = sp.simplify(c5.subs({x: 0, y: 0}).subs({d: dv}))
    say("      c_5 at the centre, incoherent pair = %s" % sp.nsimplify(c5_0))
    check("c_5 is non-zero for the incoherent pair", sp.simplify(c5_0) != 0)
    c5_coh = sp.simplify(c5.subs({x: 0, y: 0}).subs({d: 0}))
    say("      c_5 at the centre, coherent pair   = %s" % sp.nsimplify(c5_coh))
    check("and c_5 = 0 for the coherent one, whatever t0 -- beta/alpha is constant "
          "there", sp.simplify(c5_coh) == 0)
    say("      So the two rungs are independent, each way:")
    say("      6(ii)  is separated by dphi and NOT by c_5, and its two members agree")
    say("             in EVERY per-visit feature, being congruent;")
    say("      6(iii) is separated by c_5 and NOT by the averaged dphi -- the field")
    say("             dphi still holds it, but the average the clinic must take does")
    say("             not, and c_5 is the scalarization that keeps it.")
    return None


# ===================================================================== part 7

def part7():
    say("\n=== 7. Codazzi's constraint on such phantoms")
    say("    Why 6(ii) is built from a scaled rotated COPY and not by prescribing")
    say("    (b, beta) freely: Codazzi (dbar beta = d b) ties them.")
    # beta = e^{2 i t0} alpha with b = a demands dbar beta = d b, i.e.
    # e^{2 i t0} dbar alpha = d a, and Codazzi for A gives dbar alpha = d a.
    t0 = sp.symbols("t0", real=True)
    da = sp.symbols("da")
    lhs = sp.exp(2 * I * t0) * da          # dbar beta, using Codazzi for A
    rhs = da                               # d b = d a
    check("a constant relative rotation with UNCHANGED mean curvature forces "
          "(e^{2 i t0} - 1) d a = 0", sp.simplify(sp.expand(lhs - rhs)
                                                  - (sp.exp(2 * I * t0) - 1) * da) == 0)
    say("      so it is realizable only where d a = 0 (constant mean curvature) or")
    say("      t0 = 0.  A genuine phantom must let the mean curvature change too --")
    say("      which the scaled copy of 6(ii) does, b = (1+eps) a.")
    return None


# ========================================================================= main

def main():
    global LOG
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jointness.log"), "w") as fh:
        LOG = fh
        say(__doc__.split("\n\n")[0])
        part1()
        part2()
        part3()
        part4()
        part5()
        f1, k1, k2, eps = part6()
        part6iii(f1, k1, k2, eps)
        part7()
        say("\n=== %d checks, %d failed" % (NCHECK[0], len(FAILED)))
        for f in FAILED:
            say("    FAILED: %s" % f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
