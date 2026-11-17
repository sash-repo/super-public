"""The open half of Problem 3.9: the harmonics of (3.x) named, and Codazzi applied.

H. Andersen proposed the route that closes what Problem
3.9 leaves open -- not the computation (that is Statement 3.13 = \\eqref{prop39}) but
its second sentence: is the answer a new invariant of a pair of surfaces, or a
combination of the classical ones?

Her route, and what this script decides about it:

  Part 1.  Expand \\eqref{prop39} in the fiber variable.  The harmonics being 1, 3, 5,
           the answer is three coefficient fields on Sigma.  Computed here EXACTLY,
           symbolically in (a, alpha) and (b, beta) and their Wirtinger derivatives.
  Part 2.  Her c_1, c_3, c_5, checked term by term against Part 1.
  Part 3.  The same three coefficients read off the HONEST operator -- kGz solved for
           from invariance in the Darboux chart (prop39.py, its cached solution) and
           applied to two metric lifts -- so that the c_n are anchored to the primary
           computation and not merely to the formula printed in the text.
  Part 4.  Codazzi.  (a) For a genuine surface, nabla II IS totally symmetric -- the
           equation, verified from g and II of a graph with Christoffels, not quoted.
           (b) Total symmetry of the 3-jet implies her identity  dbar alpha = d a.
  Part 5.  What the identity does to the three coefficients: every dbar of a Hopf part
           is eliminated, and the answer is assembled from (H, alpha, dH, d alpha) of
           each surface.  Two terms of c_1 merge under it.
  Part 6.  c_5 as the Wronskian of the two Hopf differentials: the weight, and the
           transvectant it is.

Conventions, hers.  a = tr(A)/2, alpha = (A11 - A22)/2 - i A12, so
kappa_A = a + Re(alpha e^{2 i vartheta}); d = (d_x - i d_y)/2, dbar = (d_x + i d_y)/2;
z = e^{i vartheta}; the metric frame is (d_vartheta, nabla_v, nabla_{Jv}) with
v = (cos, sin) and Jv = (-sin, cos), which is the frame of \\eqref{prop39} -- Part 3
is what proves that this is the same frame prop39.py uses.

Nothing is asserted below that the run does not check.
"""

import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


z = sp.Symbol("z")
I = sp.I

# ===================================================================== part 1
# The symbolic Fourier expansion of (3.x) in the metric frame.

# Each surface contributes a real trace part and a complex Hopf part, each with its
# two first derivatives.  Everything below is built from these and nothing else.
def datum(tag):
    a = sp.Symbol("%s" % tag, real=True)
    ax = sp.Symbol("%s_x" % tag, real=True)
    ay = sp.Symbol("%s_y" % tag, real=True)
    h1 = sp.Symbol("%s1" % tag.upper(), real=True)
    h2 = sp.Symbol("%s2" % tag.upper(), real=True)
    h1x = sp.Symbol("%s1_x" % tag.upper(), real=True)
    h1y = sp.Symbol("%s1_y" % tag.upper(), real=True)
    h2x = sp.Symbol("%s2_x" % tag.upper(), real=True)
    h2y = sp.Symbol("%s2_y" % tag.upper(), real=True)
    al = h1 + I * h2
    alx = h1x + I * h2x
    aly = h1y + I * h2y
    return {
        "a": a, "ax": ax, "ay": ay,
        "al": al, "alx": alx, "aly": aly,
        # the Wirtinger derivatives
        "da": (ax - I * ay) / 2, "dba": (ax + I * ay) / 2,
        "dal": (alx - I * aly) / 2, "dbal": (alx + I * aly) / 2,
        "dalbar": (sp.conjugate(alx) - I * sp.conjugate(aly)) / 2,
        "dbalbar": (sp.conjugate(alx) + I * sp.conjugate(aly)) / 2,
        "albar": sp.conjugate(al),
        # the underlying real symbols, in a fixed order: substitutions are made on
        # THESE and never on the expressions built from them
        "syms": (a, ax, ay, h1, h2, h1x, h1y, h2x, h2y),
    }


def kappa(D):
    """kappa_A = a + Re(alpha e^{2 i vartheta}) as a Laurent polynomial in z."""
    return D["a"] + (D["al"] * z ** 2 + D["albar"] / z ** 2) / 2


def d_kappa(D):
    """d(kappa_A), the holomorphic Wirtinger derivative of the COEFFICIENTS."""
    return D["da"] + (D["dal"] * z ** 2 + D["dalbar"] / z ** 2) / 2


def db_kappa(D):
    """dbar(kappa_A)."""
    return D["dba"] + (D["dbal"] * z ** 2 + D["dbalbar"] / z ** 2) / 2


def dth(D):
    """d_vartheta kappa_A."""
    return I * D["al"] * z ** 2 - I * D["albar"] / z ** 2


def nabla_v(D):
    """nabla_v = e^{i vartheta} d + e^{-i vartheta} dbar."""
    return z * d_kappa(D) + db_kappa(D) / z


def nabla_Jv(D):
    """nabla_{Jv} = i(e^{i vartheta} d - e^{-i vartheta} dbar), Jv = (-sin, cos)."""
    return I * (z * d_kappa(D) - db_kappa(D) / z)


def laurent(expr):
    """The Laurent coefficients in z of an expression built from the above."""
    e = sp.expand(sp.together(expr) * z ** 8)
    p = sp.Poly(e, z)
    out = {}
    for mon, co in p.as_dict().items():
        n = int(mon[0]) - 8
        co = sp.expand(co)
        if co != 0:
            out[n] = sp.simplify(co)
    return out


def part1():
    say("\n=== 1. (3.x) expanded in the fiber variable, exactly")
    A, B = datum("a"), datum("b")
    kA, kB = kappa(A), kappa(B)
    # \eqref{prop39} read in the metric trivialization: the factor 2c is stripped.
    L = sp.expand(
        dth(A) * nabla_v(B) - nabla_v(A) * dth(B)
        - nabla_Jv(A) * kB + kA * nabla_Jv(B)
    )
    co = laurent(L)
    ns = sorted(co)
    say("  harmonics present : %s" % ns)
    check("the harmonics are exactly +-1, +-3, +-5 -- odd, none above 5, "
          "no mean (Statement 3.13 item 5)",
          ns == [-5, -3, -1, 1, 3, 5])
    # reality: the answer is a real expression, so c_{-n} = conj(c_n)
    real = all(sp.simplify(co[-n] - sp.conjugate(co[n])) == 0 for n in (1, 3, 5))
    check("c_{-n} = conj(c_n) for n = 1,3,5 -- the answer is real, so the three "
          "coefficients carry all of it", real)
    for n in (5, 3, 1):
        say("  c_%d = %s" % (n, sp.simplify(sp.expand(co[n]))))
    return A, B, co


# ===================================================================== part 2
# Her three formulas.

def her_coefficients(A, B):
    """c_1, c_3, c_5 exactly as given by H. Andersen."""
    a, al, da, dal, dbal = A["a"], A["al"], A["da"], A["dal"], A["dbal"]
    b, be, db, dbe, dbbe = B["a"], B["al"], B["da"], B["dal"], B["dbal"]
    dba, dbb = A["dba"], B["dba"]
    albar, bebar = A["albar"], B["albar"]
    dalbar, dbebar = A["dalbar"], B["dalbar"]
    c5 = sp.Rational(3, 4) * I * (al * dbe - be * dal)
    c3 = (sp.Rational(3, 2) * I * (al * db - be * da)
          + sp.Rational(1, 4) * I * (al * dbbe - be * dbal)
          + sp.Rational(1, 2) * I * (a * dbe - b * dal))
    c1 = (I * (a * db - b * da)
          - sp.Rational(1, 2) * I * (a * dbbe - b * dbal)
          + sp.Rational(1, 2) * I * (al * dbb - be * dba)
          + sp.Rational(3, 4) * I * (al * dbebar - be * dalbar)
          - sp.Rational(1, 4) * I * (albar * dbe - bebar * dal))
    return {1: c1, 3: c3, 5: c5}


def part2(A, B, co):
    say("\n=== 2. her c_1, c_3, c_5, checked against part 1")
    hers = her_coefficients(A, B)
    for n in (5, 3, 1):
        d = sp.simplify(sp.expand(co[n] - hers[n]))
        check("c_%d as given by H. Andersen is exactly the coefficient of z^%d" % (n, n),
              d == 0)
        if d != 0:
            say("      difference : %s" % d)
    # antisymmetry, term by term: the exchange A <-> B, made on the real symbols
    sw = {}
    for sA, sB in zip(A["syms"], B["syms"]):
        sw[sA] = sB
        sw[sB] = sA
    for n in (5, 3, 1):
        e = sp.expand(hers[n])
        swapped = e.subs(sw, simultaneous=True)
        check("c_%d is antisymmetric under the exchange of the two surfaces" % n,
              sp.simplify(sp.expand(e + swapped)) == 0)
    return hers


def part2b():
    say("\n=== 2b. each c_n is a spin-n object, term by term")
    say("    (the weight: a has 0, alpha has 2, conj lowers by 4, d raises by 1 and")
    say("     dbar lowers by 1 -- her 'sections of l^{tensor n}, n = 1,3,5')")
    t = sp.Symbol("t")                       # the bookkeeping parameter of the weight

    def graded(tag):
        s = lambda nm, w: sp.Symbol("%s%s" % (tag, nm)) * t ** w
        return {"a": s("a", 0), "da": s("da", 1), "dba": s("dba", -1),
                "al": s("al", 2), "dal": s("dal", 3), "dbal": s("dbal", 1),
                "albar": s("albar", -2), "dalbar": s("dalbar", -1),
                "dbalbar": s("dbalbar", -3)}

    G = her_coefficients(graded("A"), graded("B"))
    ok = True
    for n in (5, 3, 1):
        p = sp.Poly(sp.expand(G[n]), t)
        degs = sorted({int(mon[0]) for mon in p.as_dict()})
        say("  c_%d : the weights occurring are %s" % (n, degs))
        ok = ok and degs == [n]
    check("every term of every c_n has the weight of its own harmonic -- so the three "
          "coefficients are genuine spin-1, spin-3 and spin-5 fields on Sigma", ok)


# ===================================================================== part 3
# The same three coefficients, read off the honest operator.

def part3(hers):
    say("\n=== 3. the same coefficients read off the HONEST kGz")
    say("    (the operator solved for from invariance in the Darboux chart of")
    say("     PT^*R^2, applied to two metric lifts -- prop39.py's cached solution)")
    import prop39 as P

    op0, op2 = P.part2()
    terms, vec = op2
    x, y, m, c = P.x, P.y, P.m, sp.Symbol("c")

    def wirt(u):
        """(u, d u, dbar u) at the origin for a polynomial u(x,y)."""
        ux, uy = sp.diff(u, x), sp.diff(u, y)
        at0 = lambda e: sp.expand(e).subs({x: 0, y: 0})
        return at0(u), (at0(ux) - I * at0(uy)) / 2, (at0(ux) + I * at0(uy)) / 2

    def datum_of(jet):
        """Her (a, alpha) and their derivatives, out of the components of A."""
        A11, A12, A22 = jet
        a_, da_, dba_ = wirt((A11 + A22) / 2)
        al_, dal_, dbal_ = wirt((A11 - A22) / 2 - I * A12)
        return {"a": a_, "da": da_, "dba": dba_,
                "al": al_, "dal": dal_, "dbal": dbal_,
                "albar": sp.conjugate(al_),
                "dalbar": sp.conjugate(dbal_),     # d(conj u) = conj(dbar u)
                "dbalbar": sp.conjugate(dal_)}

    def fourier(expr, nu):
        """The Laurent coefficients in z = e^{i vartheta} of a section of L^nu given
        in the Darboux chart: the metric trivialization is f_alpha / cos^nu."""
        S, C = (z - 1 / z) / (2 * I), (z + 1 / z) / 2
        num, den = sp.fraction(sp.cancel(sp.together(expr)))
        pn, pd = sp.Poly(sp.expand(num), m), sp.Poly(sp.expand(den), m)
        K = max(pn.degree(), pd.degree())

        def homog(Pp):
            d = Pp.degree()
            return sp.expand(sum(cc * S ** (d - j) * C ** (K - d + j)
                                 for j, cc in enumerate(Pp.all_coeffs())))

        R = sp.cancel(sp.together(homog(pn) / (homog(pd) * C ** nu)) * z ** K)
        n2, d2 = sp.fraction(sp.cancel(R))
        assert sp.Poly(sp.expand(d2), z).degree() == 0, "not a Laurent polynomial"
        pol = sp.Poly(sp.expand(n2 / d2), z)
        out = {}
        for mon, cc in pol.as_dict().items():
            n = int(mon[0]) - K
            cc = sp.simplify(cc)
            if cc != 0:
                out[n] = cc
        return out

    ok_all = True
    for seed in (7, 8, 11):
        Aj, Bj = P.jets(100 + seed), P.jets(200 + seed)
        qA, _ = P.lift(Aj, c)
        qB, _ = P.lift(Bj, c)
        # the derivatives are taken by build_op; the value is then read AT THE ORIGIN,
        # which is where the datum (a, alpha, d a, d alpha) below is read too
        L = sp.together(P.build_op(terms, vec, qA, qB).subs({x: 0, y: 0})
                        ).subs(c, sp.Rational(3, 2))
        co = fourier(L, 3)
        ns = sorted(co)
        DA, DB = datum_of(Aj), datum_of(Bj)
        pred = her_coefficients(DA, DB)
        # (3.x) = 2c * sum c_n z^n + c.c., and c has been set to 3/2
        two_c = sp.Rational(3)
        good = (ns == [-5, -3, -1, 1, 3, 5])
        for n in (1, 3, 5):
            good = good and sp.simplify(co[n] - two_c * pred[n]) == 0
        ok_all = ok_all and good
        say("  jet pair %d: harmonics %s, the three coefficients %s"
            % (seed, ns, "agree" if good else "DISAGREE"))
    check("on three independent pairs of generic 2-jets the honest kGz has exactly "
          "the harmonics +-1,+-3,+-5 and her c_1, c_3, c_5 as its coefficients",
          ok_all)


# ===================================================================== part 4
# Codazzi.

def part4():
    say("\n=== 4a. the Codazzi equation, verified on a genuine surface")
    say("    (g and II of a graph, Christoffels built from g -- nothing quoted)")
    u, v = sp.symbols("u v", real=True)
    # a graph with no symmetry left in it
    f = (sp.Rational(1, 2) * u ** 2 - sp.Rational(2, 3) * u * v + sp.Rational(5, 4) * v ** 2
         + sp.Rational(1, 3) * u ** 3 - sp.Rational(3, 5) * u ** 2 * v
         + sp.Rational(7, 6) * u * v ** 3 - sp.Rational(2, 7) * v ** 4
         + sp.Rational(4, 9) * u ** 3 * v)
    X = [u, v]
    f1, f2 = sp.diff(f, u), sp.diff(f, v)
    W = sp.sqrt(1 + f1 ** 2 + f2 ** 2)
    g = sp.Matrix([[1 + f1 ** 2, f1 * f2], [f1 * f2, 1 + f2 ** 2]])
    II = sp.Matrix([[sp.diff(f, X[i], X[j]) / W for j in (0, 1)] for i in (0, 1)])
    ginv = g.inv()
    Gam = [[[sp.together(sum(ginv[k, l] * (sp.diff(g[l, i], X[j])
                                           + sp.diff(g[l, j], X[i])
                                           - sp.diff(g[i, j], X[l])) / 2
                             for l in (0, 1)))
             for j in (0, 1)] for i in (0, 1)] for k in (0, 1)]

    def nabII(i, j, k):
        e = sp.diff(II[j, k], X[i])
        for l in (0, 1):
            e -= Gam[l][i][j] * II[l, k] + Gam[l][i][k] * II[j, l]
        return sp.together(e)

    pt = {u: sp.Rational(1, 3), v: sp.Rational(-2, 5)}
    vals = {(i, j, k): sp.simplify(nabII(i, j, k).subs(pt))
            for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    sym = all(sp.simplify(vals[(i, j, k)] - vals[(j, i, k)]) == 0
              for i in (0, 1) for j in (0, 1) for k in (0, 1))
    check("nabla II is totally symmetric at a generic point of a generic graph "
          "(the Codazzi equation)", sym)
    nonzero = any(sp.simplify(vv) != 0 for vv in vals.values())
    check("and not vacuously: nabla II is non-zero there", nonzero)

    say("\n=== 4b. Codazzi in her notation:  dbar alpha = d a")
    say("    (in normal coordinates at the point, where nabla_i = d_i on the 1-jet)")
    A11x, A12x, A22x, A11y, A12y, A22y = sp.symbols(
        "A11x A12x A22x A11y A12y A22y", real=True)
    # total symmetry of d_i A_jk : d_1 A_22 = d_2 A_12 and d_2 A_11 = d_1 A_12
    rel = {A22x: A12y, A11y: A12x}
    ax, ay = (A11x + A22x) / 2, (A11y + A22y) / 2
    alx = (A11x - A22x) / 2 - I * A12x
    aly = (A11y - A22y) / 2 - I * A12y
    da = (ax - I * ay) / 2
    dbal = (alx + I * aly) / 2
    gap_free = sp.simplify(sp.expand(dbal - da))
    gap = sp.simplify(sp.expand((dbal - da).subs(rel)))
    check("dbar alpha - d a vanishes once the 3-jet is totally symmetric", gap == 0)
    check("and it does NOT vanish without Codazzi -- the identity has content",
          gap_free != 0)
    say("  without Codazzi, dbar alpha - d a = %s" % sp.simplify(gap_free))
    return rel


# ===================================================================== part 5
# What Codazzi does to the three coefficients.

def part5(A, B, hers):
    say("\n=== 5. the three coefficients after Codazzi")
    # dbar alpha = d a, resolved into its two REAL components and imposed on the
    # real symbols: with alpha = A1 + i A2 it reads
    #     A1_x - A2_y = a_x   and   A2_x + A1_y = - a_y,
    # so A1_x and A1_y are eliminated.  The conjugate identity d(alpha-bar) = dbar a
    # is the complex conjugate of the same two equations and needs no separate work.
    cod = {}
    for D in (A, B):
        a_, ax_, ay_, h1, h2, h1x, h1y, h2x, h2y = D["syms"]
        cod[h1x] = ax_ + h2y
        cod[h1y] = -ay_ - h2x
    for D in (A, B):
        check("the substitution really imposes dbar alpha = d a on that surface",
              sp.simplify(sp.expand((D["dbal"] - D["da"]).subs(cod))) == 0)
    red = {n: sp.simplify(sp.expand(sp.expand(hers[n]).subs(cod, simultaneous=True)))
           for n in (1, 3, 5)}
    a, al, da, dal, dba = A["a"], A["al"], A["da"], A["dal"], A["dba"]
    b, be, db, dbe, dbb = B["a"], B["al"], B["da"], B["dal"], B["dba"]
    albar, bebar = A["albar"], B["albar"]

    # the reduced shapes are written in the SAME data, so they are read under the
    # same substitution -- otherwise the two sides speak of different variables
    sub = lambda e: sp.expand(sp.expand(e).subs(cod, simultaneous=True))
    want5 = sub(sp.Rational(3, 4) * I * (al * dbe - be * dal))
    want3 = sub(sp.Rational(7, 4) * I * (al * db - be * da)
                + sp.Rational(1, 2) * I * (a * dbe - b * dal))
    want1 = sub(sp.Rational(1, 2) * I * (a * db - b * da)
                + sp.Rational(5, 4) * I * (al * dbb - be * dba)
                - sp.Rational(1, 4) * I * (albar * dbe - bebar * dal))
    check("c_5 is unchanged by Codazzi (it carries no dbar at all)",
          sp.simplify(red[5] - want5) == 0)
    check("c_3 collapses to (7i/4)(alpha d b - beta d a) + (i/2)(a d beta - b d alpha)",
          sp.simplify(red[3] - want3) == 0)
    check("c_1 collapses to (i/2)(a d b - b d a) + (5i/4)(alpha dbar b - beta dbar a) "
          "- (i/4)(alpha-bar d beta - beta-bar d alpha)",
          sp.simplify(red[1] - want1) == 0)
    say("  two terms of c_1 merge under Codazzi: (i/2)(alpha dbar b - beta dbar a)")
    say("  and (3i/4)(alpha d beta-bar - beta d alpha-bar) become the same object,")
    say("  since d(beta-bar) = conj(dbar beta) = conj(d b) = dbar b; whence 5i/4.")

    # The closure claim, counted rather than asserted.  Per surface the answer is a
    # priori a function of NINE real numbers -- H, dH (2), alpha (2), d alpha (2),
    # dbar alpha (2), the last four sharing their components -- and the question is
    # how many survive.  Codazzi imposes two relations, and 9 - 2 = 7 is exactly the
    # size of (H, alpha, dH, d alpha): 1 + 2 + 2 + 2.
    used = set()
    for n in (1, 3, 5):
        used |= sp.expand(red[n]).free_symbols
    for tag, D in (("A", A), ("B", B)):
        nine = set(D["syms"])
        left = sorted(str(s) for s in (used & nine))
        check("surface %s: the answer is left with %d of its 9 real parameters, "
              "which is the size of (H, alpha, dH, d alpha)" % (tag, len(left)),
              len(left) == 7)
        say("  surface %s keeps: %s" % (tag, left))
    check("and the two eliminated ones are exactly the components of dbar alpha "
          "that Codazzi determines",
          all(str(s) not in used for D in (A, B)
              for s in (D["syms"][5], D["syms"][6])))
    say("  (H and dH are three real numbers, alpha and d alpha four; dbar H is")
    say("   conj(d H) and dbar alpha is d H, so neither carries anything new.)")
    return red, cod


# ===================================================================== part 6
# c_5 as the Wronskian of the two Hopf differentials.

def part6(A, B, red, cod):
    say("\n=== 6. c_5 is the Wronskian of the two Hopf differentials")
    al, dal = A["al"], A["dal"]
    be, dbe = B["al"], B["dal"]
    # read under the same substitution as red[5] -- c_5 carries no dbar, but
    # its d alpha does have components that Codazzi re-expresses
    W = sp.expand((al * dbe - be * dal).subs(cod, simultaneous=True))
    check("c_5 = (3i/4) * (alpha d beta - beta d alpha), the Wronskian",
          sp.simplify(red[5] - sp.Rational(3, 4) * I * W) == 0)
    # proportionality, imposed on the real symbols: beta = 2 alpha as a FIELD means
    # the derivatives follow, beta = 2 alpha at the point alone does not
    raw = al * dbe - be * dal
    aS, bS = A["syms"], B["syms"]
    field = {bS[j]: 2 * aS[j] for j in (3, 4, 5, 6, 7, 8)}
    point = {bS[j]: 2 * aS[j] for j in (3, 4)}
    check("the Wronskian vanishes when the two Hopf differentials are proportional "
          "with a constant ratio",
          sp.simplify(sp.expand(raw.subs(field, simultaneous=True))) == 0)
    check("and does not vanish when the ratio is not constant -- it measures exactly "
          "the failure of proportionality",
          sp.simplify(sp.expand(raw.subs(point, simultaneous=True))) != 0)
    # the weight: alpha is a section of K^2 (a quadratic differential), so
    # f d g - g d f with f, g of weight k is of weight 2k+1 = 5.
    k = sp.Symbol("k")
    say("  weights: alpha, beta are quadratic differentials (weight 2); for two")
    say("  sections of weight k the combination f d g - g d f has weight 2k+1,")
    say("  which is 5 -- and z^5 is what it multiplies.  The check is the")
    say("  transformation law under a holomorphic change of the coordinate.")
    w = sp.Symbol("w")
    h = sp.Function("h")           # z = h(w), a change of the holomorphic coordinate
    F = sp.Function("F")
    G = sp.Function("G")
    hp = sp.diff(h(w), w)
    # f, g of weight 2 pull back as F(h) h'^2
    ff, gg = F(h(w)) * hp ** 2, G(h(w)) * hp ** 2
    lhs = sp.expand(sp.simplify(ff * sp.diff(gg, w) - gg * sp.diff(ff, w)))
    rhs = sp.expand(sp.simplify(
        ((F(h(w)) * sp.Subs(sp.Derivative(G(sp.Symbol("t")), sp.Symbol("t")),
                            sp.Symbol("t"), h(w))
          - G(h(w)) * sp.Subs(sp.Derivative(F(sp.Symbol("t")), sp.Symbol("t")),
                              sp.Symbol("t"), h(w))) * hp ** 5).doit()))
    check("f d g - g d f transforms with h'^5, so it is a section of K^5 -- the "
          "first transvectant of two quadratic differentials",
          sp.simplify(sp.expand(lhs - rhs)) == 0)


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "harm39.log"), "w")
    A, B, co = part1()
    hers = part2(A, B, co)
    part2b()
    part3(hers)
    part4()
    red, cod = part5(A, B, hers)
    part6(A, B, red, cod)
    say("\n%d/%d checks pass" % (CHECKS[1], CHECKS[0]))
    LOG.close()


if __name__ == "__main__":
    main()
