"""Problem 3.9 evaluated in the flat model: what kGz(Lam_g A, Lam_g B) actually is.

H. Andersen's notes left Problem 3.9 in this
state: the shortcut through the census of Subsec. 3.10 is closed (the lift is not a
k(3)-morphism), and the answer is bounded (a scalar of weight 2 has 15 coefficients).
Neither settles the Problem's own question -- whether the answer is a combination of
the classical invariants or something new -- and her own verdict "Proposition 3.9b is
false" is a verdict on a PROOF, not on the claim it was meant to prove.

This script decides the claim, in the one case in which the obstacle named in Problem
3.9 is absent: g flat.  Then the Levi-Civita connection is the flat one, the terms of
order 3 in g drop out, and the operator can simply be applied.

  Part 1.  The contact machinery on PT^*R^2 in the Darboux chart, derived not quoted.
  Part 2.  The invariant bilinear operators T(2,0) x T(2,0) -> F_nu of weighted order
           0 and 2, SOLVED for in this chart -- an independent re-derivation of the
           census of [LeLoGz, Part II] in a second coordinate system.
  Part 3.  The metric lift Lam_g in the Darboux frame (the g-frame components are
           (c kappa, 0, kappa), which is the definition; the conversion is geometry).
  Part 4.  kGz(Lam_g A, Lam_g B), evaluated.
  Part 5.  Is it P kappa_A kappa_B + Q {kappa_A, kappa_B}?  -- the claim of the
           withdrawn Proposition 3.9b, tested in its most generous reading.
  Part 6.  What it is instead, and its harmonic content.

Conventions.  Sigma = R^2 with the flat metric.  PT^*Sigma has the Darboux chart
(x,y,m), the contact element at (x,y) being the line of slope m, and

    alpha = dy - m dx,   e_1 = d_m,   e_2 = d_x + m d_y,   R = d_y = [e_1,e_2],

so (e_1,e_2,R) is a Heisenberg frame and D = ker alpha = <e_1,e_2>; d pi(e_2) = v,
v = (1,m) spanning the line.  The angle chart is m = tan(theta), v_theta = v/s with
s = sqrt(1+m^2), and the metric frame is e_1^th = d_theta = s^2 e_1, e_2^th = e_2/s.

Nothing is asserted below that the run does not check.
"""

import os
import sys
from itertools import product

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
KEYS = [(1, 1), (1, 2), (2, 2)]          # the components of a symmetric 2-form on D


# ===================================================================== part 1
# The chart, the contact fields, the frame.

_KF_CACHE = {}


def Kf(f):
    """The contact field of the generating function f, DERIVED from alpha(K)=f and
    L_K alpha = rho alpha.  Returned as (X^x, X^y, X^m).  Memoized: the derivation
    is a solve, and the solve is the same every time."""
    key = sp.srepr(sp.sympify(f))
    if key in _KF_CACHE:
        return _KF_CACHE[key]
    a_, b_, c_ = sp.symbols("a_ b_ c_")
    fx, fy, fm = sp.diff(f, x), sp.diff(f, y), sp.diff(f, m)
    # L_K alpha = d(alpha(K)) + i_K d(alpha),  d alpha = dx ^ dm
    #           = (f_x - c_) dx + f_y dy + (f_m + a_) dm  =!=  rho (dy - m dx)
    sol = sp.solve([sp.Eq(fx - c_, -m * fy), sp.Eq(fm + a_, 0), sp.Eq(b_ - m * a_, f)],
                   [a_, b_, c_], dict=True)[0]
    out = (sp.expand(sol[a_]), sp.expand(sol[b_]), sp.expand(sol[c_]))
    _KF_CACHE[key] = out
    return out


def rho(f):
    """L_{K_f} alpha = rho(f) alpha."""
    return sp.diff(f, y)


apply_v = lambda V, u: sp.expand(V[0] * sp.diff(u, x) + V[1] * sp.diff(u, y)
                                 + V[2] * sp.diff(u, m))
brk_v = lambda V, W: tuple(sp.expand(apply_v(V, W[i]) - apply_v(W, V[i])) for i in range(3))

E1 = (sp.S(0), sp.S(0), sp.S(1))                 # d_m
E2 = (sp.S(1), m, sp.S(0))                       # d_x + m d_y
RE = (sp.S(0), sp.S(1), sp.S(0))                 # d_y
FR = {1: E1, 2: E2}

# a vector in the frame (e_1, e_2, R)
inframe = lambda V: (V[2], V[0], sp.expand(V[1] - m * V[0]))

e1 = lambda u: sp.diff(u, m)
e2 = lambda u: sp.expand(sp.diff(u, x) + m * sp.diff(u, y))
Rd = lambda u: sp.diff(u, y)
WORD = {"1": lambda u: u, "e1": e1, "e2": e2, "R": Rd,
        "e1e1": lambda u: e1(e1(u)), "e1e2": lambda u: e1(e2(u)),
        "e2e2": lambda u: e2(e2(u))}
ORDER = {"1": 0, "e1": 1, "e2": 1, "R": 2, "e1e1": 2, "e1e2": 2, "e2e2": 2}


def part1():
    say("\n=== 1. the chart (x,y,m), alpha = dy - m dx")
    f = sp.Function("f")(x, y, m)
    K = Kf(f)
    say("  K_f = (%s) d_x + (%s) d_y + (%s) d_m" % K)
    al = lambda V: sp.expand(V[1] - m * V[0])
    check("alpha(K_f) = f", sp.simplify(al(K) - f) == 0)
    check("alpha(e_1) = alpha(e_2) = 0", al(E1) == 0 and al(E2) == 0)
    check("[e_1,e_2] = R", brk_v(E1, E2) == RE)
    check("[e_1,R] = [e_2,R] = 0",
          brk_v(E1, RE) == (0, 0, 0) and brk_v(E2, RE) == (0, 0, 0))
    # the Heisenberg subalgebra k_{-2} + k_{-1} commutes with the frame
    for gen in [sp.S(1), x, m]:
        V = Kf(gen)
        ok = all(brk_v(V, FR[i]) == (0, 0, 0) for i in (1, 2))
        check("K_{%s} commutes with the frame (so invariant operators have constant "
              "coefficients in it)" % gen, ok)
    # [K_f, K_g] = K_{f,g}
    f1 = 3 * x ** 2 - 2 * x * y + m ** 2 * y + m
    g1 = x * y - m ** 3 + 2 * y ** 2
    cb = al(brk_v(Kf(f1), Kf(g1)))
    check("[K_f,K_g] = K_{f,g}",
          all(sp.simplify(brk_v(Kf(f1), Kf(g1))[i] - Kf(cb)[i]) == 0 for i in range(3)))
    # d pi (e_2) = v = (1,m):  the line of slope m
    check("d pi (e_1) = 0 (e_1 is vertical)", (E1[0], E1[1]) == (0, 0))
    check("d pi (e_2) = (1,m), the line of slope m", (E2[0], E2[1]) == (1, m))


# ===================================================================== part 2
# The modules, and the invariant bilinear operators, solved for.

def L_F(f, u, lam):
    """The action of k(3) on F_lam."""
    return sp.expand(apply_v(Kf(f), u) + lam * rho(f) * u)


_C_CACHE = {}


def Cmat(f):
    """[K_f, e_a] = sum_b C[a][b] e_b  (the Reeb component must vanish)."""
    key = sp.srepr(sp.sympify(f))
    if key in _C_CACHE:
        return _C_CACHE[key]
    K = Kf(f)
    C = {}
    for i in (1, 2):
        c1, c2, cR = inframe(brk_v(K, FR[i]))
        assert sp.simplify(cR) == 0, "the bracket leaves D"
        C[i] = {1: sp.expand(c1), 2: sp.expand(c2)}
    _C_CACHE[key] = C
    return C


def L_T2(f, q, lam):
    """The action of k(3) on T(2,lam), components in the frame (e_1,e_2)."""
    C = Cmat(f)
    Q = lambda i, j: q[(min(i, j), max(i, j))]
    out = {}
    for (i, j) in KEYS:
        val = apply_v(Kf(f), Q(i, j)) + lam * rho(f) * Q(i, j)
        val -= sum(C[i][k] * Q(k, j) for k in (1, 2))
        val -= sum(C[j][k] * Q(i, k) for k in (1, 2))
        out[(i, j)] = sp.expand(val)
    return out


def rpoly(seed, deg=3):
    """A pseudo-random polynomial of the given degree, exact rationals."""
    import random
    r = random.Random(seed)
    out = sp.S(0)
    for a in range(deg + 1):
        for b in range(deg + 1):
            for c in range(deg + 1):
                if a + b + c <= deg:
                    out += sp.Rational(r.randint(-4, 4), r.randint(1, 3)) * x ** a * y ** b * m ** c
    return sp.expand(out)


def part2_axioms():
    say("\n=== 2a. the module axioms")
    f1, g1 = rpoly(1, 2), rpoly(2, 2)
    al = lambda V: sp.expand(V[1] - m * V[0])
    cb = al(brk_v(Kf(f1), Kf(g1)))
    lam = sp.Rational(5, 3)
    u1 = rpoly(3, 3)
    lhs = sp.expand(L_F(f1, L_F(g1, u1, lam), lam) - L_F(g1, L_F(f1, u1, lam), lam))
    check("F_lam is a k(3)-module", sp.simplify(lhs - L_F(cb, u1, lam)) == 0)
    q1 = {k: rpoly(10 + i, 3) for i, k in enumerate(KEYS)}
    A_ = L_T2(f1, L_T2(g1, q1, lam), lam)
    B_ = L_T2(g1, L_T2(f1, q1, lam), lam)
    C_ = L_T2(cb, q1, lam)
    check("T(2,lam) is a k(3)-module",
          all(sp.simplify(A_[k] - B_[k] - C_[k]) == 0 for k in KEYS))


def ansatz_terms(W):
    """The word pairs of total weighted order W, with a reduced word basis
    (e2e1 = e1e2 - R, so it is not needed)."""
    words = [w for w in WORD]
    return [(w1, w2, kf, kg)
            for w1 in words for w2 in words
            if ORDER[w1] + ORDER[w2] == W
            for kf in KEYS for kg in KEYS]


def build_op(terms, coefs, F, G):
    out = sp.S(0)
    for t, c in zip(terms, coefs):
        if c == 0:
            continue
        w1, w2, kf, kg = t
        out += c * WORD[w1](F[kf]) * WORD[w2](G[kg])
    return sp.expand(out)


GENS_SOLVE = [x, y, m, x * m, y * m, m ** 2, x ** 2, x * y, m ** 3, x * m ** 2,
              y ** 2, x ** 2 * m]
GENS_CHECK = [x * y ** 2, y * m ** 2, x ** 2 * y, x * y * m, y ** 3, m ** 4,
              x ** 3, x * m ** 3, y ** 2 * m, x ** 2 * m ** 2]


def _residual_rows(terms, lamF, lamG, gens, nsamp):
    """The invariance condition is LINEAR in the unknown coefficients AND affine in
    the target weight nu, so the whole scan needs two matrices and no re-computation:
        residual(nu) = base + nu * rhopart.
    Assembled column by column, one residual per term, no symbolic unknowns."""
    base, rhop = {}, {}
    for s in range(nsamp):
        F = {k: rpoly(100 + 10 * s + i, 3) for i, k in enumerate(KEYS)}
        G = {k: rpoly(200 + 10 * s + i, 3) for i, k in enumerate(KEYS)}
        for f in gens:
            LF, LG = L_T2(f, F, lamF), L_T2(f, G, lamG)
            rf = rho(f)
            for j, t in enumerate(terms):
                w1, w2, kf, kg = t
                val = sp.expand(WORD[w1](F[kf]) * WORD[w2](G[kg]))
                res = sp.expand(apply_v(Kf(f), val)
                                - WORD[w1](LF[kf]) * WORD[w2](G[kg])
                                - WORD[w1](F[kf]) * WORD[w2](LG[kg]))
                for tgt, ex in ((base, res), (rhop, sp.expand(rf * val))):
                    if ex == 0:
                        continue
                    for mon, co in sp.Poly(ex, x, y, m).as_dict().items():
                        tgt.setdefault((s, str(f), mon), {})[j] = sp.Rational(co)
    return base, rhop


def _as_fractions(rows, n):
    """The row dicts once, as sparse Fraction rows -- built one time, not per nu."""
    from fractions import Fraction as Fr
    out = {}
    for k, r in rows.items():
        out[k] = {j: Fr(int(v.p), int(v.q)) for j, v in
                  ((j, sp.Rational(c)) for j, c in r.items()) if v != 0}
    return out


def _nullspace_of(base, rhop, nu, n):
    from fractions import Fraction as Fr
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from exactlin import nullspace                                   # noqa: E402
    M = []
    for k in set(base) | set(rhop):
        b, r = base.get(k, {}), rhop.get(k, {})
        idx = set(b) | set(r)
        if not idx:
            continue
        row = [Fr(0)] * n
        for j in idx:
            row[j] = b.get(j, Fr(0)) + nu * r.get(j, Fr(0))
        if any(row):
            M.append(row)
    if not M:
        return [[sp.S(1) if i == j else sp.S(0) for i in range(n)] for j in range(n)]
    return [[sp.Rational(v.numerator, v.denominator) for v in vec] for vec in nullspace(M)]


def solve_invariants(W, lamF, lamG, nus, nsamp=2, gens=None):
    """All bilinear invariant operators T(2,lamF) x T(2,lamG) -> F_nu of weighted
    order W, for each nu in nus: {nu: [coefficient vectors]}."""
    terms = ansatz_terms(W)
    base, rhop = _residual_rows(terms, lamF, lamG, gens or GENS_SOLVE, nsamp)
    n = len(terms)
    base, rhop = _as_fractions(base, n), _as_fractions(rhop, n)
    say("    [W = %d: %d unknowns, %d conditions]" % (W, n, len(set(base) | set(rhop))))
    return terms, {nu: _nullspace_of(base, rhop, nu, n) for nu in nus}


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prop39_ops.json")


def part2():
    say("\n=== 2b. the invariant bilinear operators T(2,0) x T(2,0) -> F_nu, solved for")
    import json
    import os
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            raw = json.load(fh)
        say("  [read back from %s -- the solve is the same every time]"
            % os.path.basename(CACHE))
        out = {}
        for key, (tm, vec) in raw.items():
            out[key] = ([tuple(t[:2]) + (tuple(t[2]), tuple(t[3])) for t in tm],
                        [sp.Rational(v) for v in vec])
        return out["0"], out["2"]
    found = {}
    for W in (0, 1, 2):
        terms, res = solve_invariants(W, 0, 0, list(range(0, 5)))
        for nu, bas in sorted(res.items()):
            if bas:
                say("  W = %d, nu = %d: a space of dimension %d" % (W, nu, len(bas)))
                found[(W, nu)] = [(terms, v) for v in bas]
    check("at W = 0 the only invariant is the fiber contraction, and it lands in F_2",
          [k for k in found if k[0] == 0] == [(0, 2)] and len(found[(0, 2)]) == 1)
    check("at W = 1 there is nothing (an odd order gives no invariant here)",
          [k for k in found if k[0] == 1] == [])
    check("at W = 2 there is exactly one invariant, and it lands in F_3",
          [k for k in found if k[0] == 2] == [(2, 3)] and len(found[(2, 3)]) == 1)
    op0, op2 = found[(0, 2)][0], found[(2, 3)][0]
    with open(CACHE, "w") as fh:
        json.dump({"0": [[list(t) for t in op0[0]], [str(v) for v in op0[1]]],
                   "2": [[list(t) for t in op2[0]], [str(v) for v in op2[1]]]}, fh)
    return op0, op2


def show_op(name, op, maxterms=40):
    terms, vec = op
    say("  %s =" % name)
    shown = 0
    for t, c in zip(terms, vec):
        if c == 0:
            continue
        shown += 1
        if shown <= maxterms:
            w1, w2, kf, kg = t
            say("      %+s * (%s F_%d%d)(%s G_%d%d)"
                % (sp.nsimplify(c), w1, kf[0], kf[1], w2, kg[0], kg[1]))
    say("    (%d non-zero terms)" % shown)


def verify_invariance(op, lamF, lamG, nu, label):
    """The operator found on six generators is re-checked on fourteen, with fresh
    data: the solve is an ansatz, this is the proof."""
    terms, vec = op
    ok = True
    for s in (0, 1):
        F = {k: rpoly(600 + 10 * s + i, 3) for i, k in enumerate(KEYS)}
        G = {k: rpoly(650 + 10 * s + i, 3) for i, k in enumerate(KEYS)}
        for f in GENS_CHECK:
            res = sp.expand(L_F(f, build_op(terms, vec, F, G), nu)
                            - build_op(terms, vec, L_T2(f, F, lamF), G)
                            - build_op(terms, vec, F, L_T2(f, G, lamG)))
            if sp.simplify(res) != 0:
                ok = False
    check("%s is invariant on 14 generators with fresh data" % label, ok)


def part2_props(op0, op2):
    say("\n=== 2c. the properties of the two operators")
    verify_invariance(op2, 0, 0, 3, "kGz")
    verify_invariance(op0, 0, 0, 2, "the order-0 contraction")
    terms, vec = op2
    F = {k: rpoly(300 + i, 3) for i, k in enumerate(KEYS)}
    G = {k: rpoly(400 + i, 3) for i, k in enumerate(KEYS)}
    val = build_op(terms, vec, F, G)
    swp = build_op(terms, vec, G, F)
    anti = sp.simplify(val + swp) == 0
    check("kGz is antisymmetric in its two arguments", anti)
    check("kGz vanishes on a repeated argument",
          sp.simplify(build_op(terms, vec, F, F)) == 0)
    t0, v0 = op0
    val0 = build_op(t0, v0, F, G)
    check("the order-0 contraction is SYMMETRIC (it is the fiber pairing of two "
          "symmetric forms)", sp.simplify(val0 - build_op(t0, v0, G, F)) == 0)
    # the pull-back locus: q11 = q12 = 0
    PB = lambda k: {(1, 1): sp.S(0), (1, 2): sp.S(0), (2, 2): k}
    kA, kB = rpoly(500, 3), rpoly(501, 3)
    check("kGz vanishes on two pull-backs (Prop. 3.6 at W=2)",
          sp.simplify(build_op(terms, vec, PB(kA), PB(kB))) == 0)
    check("the W=0 contraction vanishes on two pull-backs (Prop. 3.6 at W=0)",
          sp.simplify(build_op(t0, v0, PB(kA), PB(kB))) == 0)


# ===================================================================== part 3
# The metric lift, in the Darboux frame.

def lift(Acomp, c):
    """Lam_g(A) for the FLAT metric, components in the Darboux frame (e_1,e_2).

    The definition is the g-frame one: in the metric frame (e_1^th, e_2^th) =
    (d_theta, the horizontal lift of the UNIT vector) the components are
    (c kappa_A, 0, kappa_A).  Since e_1 = e_1^th/s^2 and e_2 = s e_2^th with
    s^2 = 1+m^2, in the Darboux frame they become (c kappa_A/s^4, 0, s^2 kappa_A);
    and kappa_A = P_A/s^2 with P_A = A11 + 2 m A12 + m^2 A22 = A(v,v), v = (1,m).
    """
    A11, A12, A22 = Acomp
    P = sp.expand(A11 + 2 * m * A12 + m ** 2 * A22)
    s2 = 1 + m ** 2
    kap = sp.together(P / s2)
    return {(1, 1): sp.together(c * P / s2 ** 3), (1, 2): sp.S(0), (2, 2): P}, kap


def part3():
    say("\n=== 3. the metric lift in the Darboux frame")
    c = sp.Symbol("c")
    A = [sp.Function("A%d" % i)(x, y) for i in (11, 12, 22)]
    q, kap = lift(A, c)
    say("  kappa_A     = %s" % kap)
    say("  q_11        = %s" % q[(1, 1)])
    say("  q_12        = %s" % q[(1, 2)])
    say("  q_22        = %s" % q[(2, 2)])
    # the horizontal component is pi^*A read in the frame: A(d pi e_2, d pi e_2)
    v = (sp.S(1), m)
    AA = sp.expand(A[0] * v[0] ** 2 + 2 * A[1] * v[0] * v[1] + A[2] * v[1] ** 2)
    check("q_22 = A(v,v) with v = d pi(e_2) -- the pull-back, no normalization",
          sp.simplify(q[(2, 2)] - AA) == 0)
    check("kappa_A = A(v_theta,v_theta) with v_theta the unit vector",
          sp.simplify(kap - AA / (1 + m ** 2)) == 0)
    # A is recovered from kappa_A: the three harmonics 0, 2 of a quadratic form
    check("A is recovered from kappa_A (nothing is lost in the avatar)",
          sp.simplify(sp.together(kap * (1 + m ** 2)) - AA) == 0)
    return c


# ===================================================================== part 4/5/6

def jets(seed, order=2):
    """A generic 2-jet of a symmetric 2-tensor field at the origin: the components
    are polynomials of degree <= order in (x,y) with pseudo-random coefficients."""
    import random
    r = random.Random(seed)
    out = []
    for _ in range(3):
        p = sp.S(0)
        for a in range(order + 1):
            for b in range(order + 1):
                if a + b <= order:
                    p += sp.Rational(r.randint(-5, 5), r.randint(1, 4)) * x ** a * y ** b
        out.append(sp.expand(p))
    return out


def bracket(F, G):
    """The invariant pairing of two curvature FUNCTIONS at lam = mu = 0:
    {F,G} = (e_1 F)(e_2 G) - (e_2 F)(e_1 G)."""
    return sp.expand(e1(F) * e2(G) - e2(F) * e1(G))


def harmonics(expr, nu):
    """The exact Fourier harmonics in the fiber angle of an expression given in the
    Darboux chart, read in the METRIC trivialization.

    A section of L^nu is f_alpha alpha^nu = f_theta alpha_theta^nu with
    alpha_theta = cos(theta) alpha, so f_theta = f_alpha / cos^nu(theta); and
    m = tan(theta), so with z = exp(i theta), sin = (z - 1/z)/2i, cos = (z + 1/z)/2
    the whole expression becomes a Laurent polynomial in z -- which is the check that
    the object is a global section over the circle of directions at a point."""
    z = sp.Symbol("z")
    S, C = (z - 1 / z) / (2 * sp.I), (z + 1 / z) / 2
    num, den = sp.fraction(sp.cancel(sp.together(expr)))
    pn, pd = sp.Poly(sp.expand(num), m), sp.Poly(sp.expand(den), m)
    K = max(pn.degree(), pd.degree())

    def homog(P):
        d = P.degree()
        return sp.expand(sum(co * S ** (d - j) * C ** (K - d + j)
                             for j, co in enumerate(P.all_coeffs())))

    R = sp.cancel(sp.together(homog(pn) / (homog(pd) * C ** nu)) * z ** K)
    n2, d2 = sp.fraction(sp.cancel(R))
    assert sp.Poly(sp.expand(d2), z).degree() == 0, "not a Laurent polynomial: %s" % R
    P = sp.Poly(sp.expand(n2 / d2), z)
    out = sorted({abs(int(mon[0]) - K) for mon, co in P.as_dict().items()
                  if sp.simplify(co) != 0})
    return out


def part456(op2, op0, c):
    say("\n=== 4. kGz(Lam_g A, Lam_g B), evaluated")
    terms, vec = op2
    A, B = jets(7), jets(8)
    qA, kA = lift(A, c)
    qB, kB = lift(B, c)
    val = sp.simplify(sp.together(build_op(terms, vec, qA, qB)))
    val0 = sp.simplify(sp.together(build_op(op0[0], op0[1], qA, qB)))
    say("  the W=0 contraction of the two lifts is, at the origin,")
    say("      %s" % sp.simplify(val0.subs({x: 0, y: 0})))
    check("the W=0 contraction of two lifts is a multiple of kappa_A kappa_B",
          sp.simplify(sp.together(val0 - sp.simplify(sp.together(
              val0 / (kA * kB))) * kA * kB)) == 0)
    check("kGz(Lam A, Lam B) is not identically zero", sp.simplify(val) != 0)
    check("kGz(Lam A, Lam A) = 0 (the answer is antisymmetric in (A,B))",
          sp.simplify(sp.together(build_op(terms, vec, qA, qA))) == 0)

    say("\n=== 5. is it  P kappa_A kappa_B + Q {kappa_A, kappa_B}?  (Prop. 3.9b)")
    say("    tested in its most generous reading: P and Q allowed to be arbitrary")
    say("    functions of the direction as well as of the point.")
    rows = []
    NS = 6
    for s_ in range(NS):
        A_, B_ = jets(700 + s_), jets(800 + s_)
        qA_, kA_ = lift(A_, c)
        qB_, kB_ = lift(B_, c)
        L = build_op(terms, vec, qA_, qB_)
        T1 = sp.expand(kA_ * kB_)
        T2 = bracket(kA_, kB_)
        rows.append((L, T1, T2))
    # at a numerical point and a numerical direction, over the samples
    worst = None
    for mv in [sp.Rational(1, 3), sp.Rational(2), sp.Rational(-3, 5)]:
        sub = {x: sp.Rational(1, 5), y: sp.Rational(-2, 7), m: mv, c: sp.Rational(3, 2)}
        M = sp.Matrix([[sp.simplify(r[0].subs(sub)), sp.simplify(r[1].subs(sub)),
                        sp.simplify(r[2].subs(sub))] for r in rows])
        rk_all, rk_T = M.rank(), M[:, 1:].rank()
        say("  m = %-6s : rank[L | T1 T2] = %d, rank[T1 T2] = %d  ->  %s"
            % (mv, rk_all, rk_T, "L is in the span" if rk_all == rk_T else
               "L is NOT in the span"))
        worst = (rk_all, rk_T) if worst is None else worst
        check("at m = %s the answer is NOT of the form P kappa kappa + Q {kappa,kappa}"
              % mv, rk_all > rk_T)

    say("\n=== 6. what it is instead")
    A_, B_ = jets(7), jets(8)
    qA_, kA_ = lift(A_, c)
    qB_, kB_ = lift(B_, c)
    L = sp.together(build_op(terms, vec, qA_, qB_))
    # The answer is antisymmetric, so it is a combination of the FIVE antisymmetric
    # bilinear blocks of weighted order <= 2 in the curvature functions, with
    # coefficients depending on the direction alone (a flat g is homogeneous).  One
    # equation per jet sample, read at the origin: a small system over Q(m).
    V, H = e1, e2
    BLOCKS = {
        "(V kA) kB - kA (V kB)": lambda a, b: V(a) * b - a * V(b),
        "(H kA) kB - kA (H kB)": lambda a, b: H(a) * b - a * H(b),
        "{kA,kB} = (V kA)(H kB) - (H kA)(V kB)": lambda a, b: V(a) * H(b) - H(a) * V(b),
        "(V^2 kA) kB - kA (V^2 kB)": lambda a, b: V(V(a)) * b - a * V(V(b)),
        "(H^2 kA) kB - kA (H^2 kB)": lambda a, b: H(H(a)) * b - a * H(H(b)),
        "(VH kA) kB - kA (VH kB)": lambda a, b: V(H(a)) * b - a * V(H(b)),
        "(R kA) kB - kA (R kB)": lambda a, b: Rd(a) * b - a * Rd(b),
    }
    names = list(BLOCKS)
    at0 = lambda e: sp.cancel(sp.together(e).subs({x: 0, y: 0}))
    rows, rhs = [], []
    for s_ in range(10):
        A2, B2 = jets(900 + s_), jets(950 + s_)
        qa, ka = lift(A2, c)
        qb, kb = lift(B2, c)
        rows.append([at0(BLOCKS[nm](ka, kb)) for nm in names])
        rhs.append(at0(build_op(terms, vec, qa, qb)))
    M, r = sp.Matrix(rows), sp.Matrix(rhs)
    aug = M.row_join(r)
    say("  rank of the seven blocks over 8 jet samples : %d" % M.rank())
    say("  rank with the answer adjoined              : %d" % aug.rank())
    inspan = M.rank() == aug.rank()
    check("the answer IS a combination of the seven antisymmetric blocks, with "
          "coefficients depending on the direction alone", inspan)
    if inspan:
        ps = sp.symbols("p0:%d" % len(names))
        sol = sp.solve([sp.Eq(sum(p * a for p, a in zip(ps, row)), b)
                        for row, b in zip(rows, rhs)], ps, dict=True)
        if sol:
            say("  the answer, exactly (m = tan(theta), s^2 = 1+m^2):")
            nonzero = []
            for p, nm in zip(ps, names):
                cf = sp.factor(sp.simplify(sol[0].get(p, p)))
                say("      %-42s  %s" % (nm, cf))
                if sp.simplify(cf) != 0:
                    nonzero.append(nm)
            check("more than the bracket occurs in it, so Prop. 3.9b's two-term "
                  "shape fails here too", len(nonzero) > 1)
    # ---- the closed form, in the METRIC frame -----------------------------
    # e_1 = d_theta / s^2,  e_2 = s * nabla_{v},  and the Reeb field of the metric
    # contact form alpha_theta = cos(theta) alpha is the horizontal lift of the
    # PERPENDICULAR direction J v, namely R_theta = s R - (m/s) e_2.  In those terms
    # the answer collapses: the two nabla_v terms outside the bracket cancel.
    s2 = 1 + m ** 2
    dth = lambda u: sp.expand(s2 * e1(u))                     # d_theta
    nab = lambda u: sp.together(e2(u) / sp.sqrt(s2))          # nabla_{v-hat}
    Rth = lambda u: sp.together(sp.sqrt(s2) * Rd(u) - m * e2(u) / sp.sqrt(s2))
    closed = sp.together(2 * c / s2 ** sp.Rational(3, 2) * (
        dth(kA_) * nab(kB_) - nab(kA_) * dth(kB_)
        - Rth(kA_) * kB_ + kA_ * Rth(kB_)))
    check("the answer equals  2c s^{-3} [ (d_theta kA)(nabla kB) - (nabla kA)"
          "(d_theta kB) - (R_theta kA) kB + kA (R_theta kB) ]",
          sp.simplify(sp.together(L - closed)) == 0)
    check("the extra term is not a multiple of the bracket: dropping it changes "
          "the answer",
          sp.simplify(sp.together(L - sp.together(
              2 * c / s2 ** sp.Rational(3, 2)
              * (dth(kA_) * nab(kB_) - nab(kA_) * dth(kB_))))) != 0)

    say("\n  harmonic content in the fiber angle theta (m = tan theta), EXACTLY:")
    for name, expr in (("kGz(Lam A, Lam B)", L),
                       ("the bracket {kappa_A,kappa_B}", bracket(kA_, kB_)),
                       ("kappa_A", kA_)):
        nu_ = {"kGz(Lam A, Lam B)": 3, "the bracket {kappa_A,kappa_B}": 1,
               "kappa_A": 0}[name]
        hs = harmonics(expr.subs({x: sp.Rational(1, 5), y: sp.Rational(-2, 7),
                                  c: sp.Rational(3, 2)}), nu_)
        say("      %-28s : %s" % (name, hs))
        if name == "kGz(Lam A, Lam B)":
            check("no harmonic above 6 in the answer (the corrected bound)",
                  hs and max(hs) <= 6)
            check("the answer is ODD and stops at 5 -- the signature of the "
                  "withdrawn prediction, which therefore holds for it after all",
                  all(h % 2 == 1 for h in hs) and max(hs) == 5)
            check("its mean over the directions vanishes (no harmonic 0)",
                  0 not in hs)
        if name == "the bracket {kappa_A,kappa_B}":
            check("the bracket has the harmonics 1, 3, 5 and no others "
                  "(Statement 3.11 stands)", hs == [1, 3, 5])
        if name == "kappa_A":
            check("a curvature function has the harmonics 0 and 2", hs == [0, 2])


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prop39.log"), "w")
    part1()
    part2_axioms()
    op0, op2 = part2()
    show_op("kGz (weighted order 2)", op2)
    show_op("the order-0 contraction", op0)
    part2_props(op0, op2)
    c = part3()
    part456(op2, op0, c)
    say("\n%d/%d checks pass" % (CHECKS[1], CHECKS[0]))
    LOG.close()


if __name__ == "__main__":
    main()
