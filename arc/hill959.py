"""Radul, pp. 40-41: the classical normalization of the Hill operator,
recomputed from Radul's own transformation law.

His n. 0.2 considers  L(phi dx^{-lam}) = (a phi'' + b phi' + c phi) dx^{mu}
and prints, for the change of variables x = g(x_1),

  L^g = g'^mu ( a^g g'^{-1} d_{x1} g'^{-1} d_{x1} + b^g g'^{-1} d_{x1} + c^g ) g'^lam   (3)

and then says: "for lam = -1/2, mu = 3/2 the conditions a(x) = 1, b(x) = 0 are
invariant with respect to the change of variable and c dx^2 changes by the
Schwartz derivative".

Two things are checked here from (3) alone, i.e. from his own print:

  1) which (lam, mu) make a = 1, b = 0 invariant -- the answer decides between
     the lam = -1/2 of his n. 0.2 and the lam = 1/2 of his n. 5.1 (p. 52) and
     of eq. (197) of the text;
  2) what c then changes by -- whether it is his S(g) of n. 0.1 or half of it,
     the row n = 0 of Table (194) of the text carrying a 1/2 that his n. 0.1 does not.

Exact, with sympy.   Run:  ../venv/bin/python hill959.py
"""
import sys
import sympy as sp

OK = BAD = 0


def check(name, ok):
    global OK, BAD
    if ok:
        OK += 1
        print("  ok   %s" % name, flush=True)
    else:
        BAD += 1
        print("  FAIL %s" % name, flush=True)


x1 = sp.Symbol('x1')
lam, mu = sp.symbols('lambda mu')
g = sp.Function('g')(x1)
phi = sp.Function('phi')(x1)

gp = sp.diff(g, x1)


def conjugate(a, b, c, lam, mu):
    """The three coefficients of L^g, from (3), read off a test function."""
    gl = gp ** lam
    inner = sp.diff(gl * phi, x1) / gp
    out = (a * sp.diff(inner, x1) / gp + b * inner + c * gl * phi)
    out = sp.expand(sp.simplify(gp ** mu * out))
    d2 = sp.Symbol('d2')
    d1 = sp.Symbol('d1')
    d0 = sp.Symbol('d0')
    out = out.subs({sp.Derivative(phi, (x1, 2)): d2,
                    sp.Derivative(phi, x1): d1, phi: d0})
    out = sp.expand(out)
    A = sp.simplify(out.coeff(d2))
    B = sp.simplify(out.coeff(d1))
    C = sp.simplify(out.coeff(d0))
    return A, B, C


def main():
    # ---- 1) the coefficients of (3) for general a, b, c ------------------
    a, b, c = sp.symbols('a b c')          # a^g, b^g, c^g: constants in x1
    A, B, C = conjugate(a, b, c, lam, mu)
    check("the d^2 coefficient of (3) is a^g g'^{lam+mu-2}",
          sp.simplify(A - a * gp ** (lam + mu - 2)) == 0)
    check("the d^1 coefficient of (3) is (2 lam - 1) a^g g'^{lam+mu-3} g'' "
          "+ b^g g'^{lam+mu-1}",
          sp.simplify(B - ((2 * lam - 1) * a * gp ** (lam + mu - 3)
                           * sp.diff(g, x1, 2)
                           + b * gp ** (lam + mu - 1))) == 0)
    want_C = (a * (lam * (lam - 2) * gp ** (lam + mu - 4) * sp.diff(g, x1, 2) ** 2
                   + lam * gp ** (lam + mu - 3) * sp.diff(g, x1, 3))
              + b * lam * sp.diff(g, x1, 2) * gp ** (lam + mu - 2)
              + c * gp ** (lam + mu))
    check("the d^0 coefficient of (3) is his own, the two b^g summands included",
          sp.simplify(C - want_C) == 0)

    # ---- 2) which (lam, mu) leave a = 1, b = 0 invariant -----------------
    A1, B1, C1 = conjugate(1, 0, c, lam, mu)
    sol = sp.solve([sp.simplify(sp.log(A1).expand(force=True)),  # lam+mu-2 = 0
                    2 * lam - 1], [lam, mu], dict=True)
    print("     solving  a_new = 1, b_new = 0  gives  %s" % sol, flush=True)
    check("the conditions a = 1, b = 0 are invariant exactly for lam = 1/2, "
          "mu = 3/2 -- NOT for the lam = -1/2 of his n. 0.2",
          len(sol) == 1 and sol[0][lam] == sp.Rational(1, 2)
          and sol[0][mu] == sp.Rational(3, 2))

    # and, explicitly, lam = -1/2 does not work
    A2, B2, C2 = conjugate(1, 0, c, sp.Rational(-1, 2), sp.Rational(3, 2))
    check("at lam = -1/2, mu = 3/2 the new b is not zero", sp.simplify(B2) != 0)
    check("   and the new a is not 1", sp.simplify(A2 - 1) != 0)

    # ---- 3) what c changes by at lam = 1/2, mu = 3/2 ---------------------
    A3, B3, C3 = conjugate(1, 0, c, sp.Rational(1, 2), sp.Rational(3, 2))
    check("at lam = 1/2, mu = 3/2 the new a is 1", sp.simplify(A3 - 1) == 0)
    check("   and the new b is 0", sp.simplify(B3) == 0)

    g2, g3 = sp.diff(g, x1, 2), sp.diff(g, x1, 3)
    Sch = g3 / gp - sp.Rational(3, 2) * g2 ** 2 / gp ** 2   # his n. 0.1 S(g)
    delta = sp.simplify(C3 - c * gp ** 2)
    print("     c_new - g'^2 c^g = %s" % sp.simplify(delta), flush=True)
    ratio = sp.simplify(delta / Sch)
    print("     (c_new - g'^2 c^g) / S(g) = %s" % ratio, flush=True)
    check("c changes by exactly HALF the S(g) of his n. 0.1, i.e. by the "
          "row n = 0 of Table (194)", sp.simplify(ratio - sp.Rational(1, 2)) == 0)

    # ---- 4) the weight: the potential is a density of weight 2 ----------
    check("the potential c is a section of F_{lam+mu} = F_2",
          sp.simplify(sp.Rational(1, 2) + sp.Rational(3, 2) - 2) == 0)

    print("\n%d ok, %d FAILED" % (OK, BAD))
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
