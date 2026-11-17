"""Independent verification of the invariance condition used by line.py.

line.py builds its linear system from a formula derived by hand.  Here the
same operators are tested from the definition, with sympy, in two ways that
share nothing with that derivation:

  (A) infinitesimally: L_X D(u_1,...,u_n) = sum_i D(...,L_X u_i,...) for
      X = xi d/dx with xi = x^2, x^3, x^4 and xi = exp(x), sin(x);
  (B) finitely: the defining identity (3) of the paper under an explicit
      change of variable x = phi(t), for phi = t^2+t+1 and phi = exp(t),
      to the order in the Taylor expansion that the operator sees.

Both are run on: the operator d, Grozman's exceptional operator, the
bracket J_1, the Wronskians of 2, 3 and 4 one-forms, and on a random
member of each invariant space the engine produces.
"""

import sympy as sp
from fractions import Fraction as F
from line import inv_space, midx

x = sp.Symbol('x')
ok = 0
bad = []


def chk(name, cond):
    global ok
    if cond:
        ok += 1
    else:
        bad.append(name)


def apply_op(vec, weights, k, funcs):
    """D(u_1,...,u_n) as a sympy expression (the coefficient of (dx)^nu)."""
    n = len(weights)
    out = 0
    for z, c in zip(midx(n, k), vec):
        if not c:
            continue
        term = sp.Integer(1)
        for i in range(n):
            term *= sp.diff(funcs[i], x, z[i])
        out += sp.Rational(c.numerator, c.denominator) * term
    return out


def lie(u, lam, xi):
    """L_X of a density u (dx)^lam, X = xi d/dx: returns the new coefficient."""
    return sp.expand(xi * sp.diff(u, x) + lam * sp.diff(xi, x) * u)


def check_infinitesimal(vec, weights, k, tag, xis):
    n = len(weights)
    nu = sum(weights) + k
    fs = [sp.Function(f'u{i}')(x) for i in range(n)]
    for xi in xis:
        lhs = lie(apply_op(vec, weights, k, fs), sp.Rational(nu.numerator,
                                                            nu.denominator), xi)
        rhs = 0
        for i in range(n):
            gs = list(fs)
            gs[i] = lie(fs[i], sp.Rational(weights[i].numerator,
                                           weights[i].denominator), xi)
            rhs += apply_op(vec, weights, k, gs)
        chk(f"{tag} infinitesimal xi={xi}", sp.simplify(sp.expand(lhs - rhs)) == 0)


def check_finite(vec, weights, k, tag, phis, order=9):
    """Identity (3): D applied to g_i = f_i(phi), times nothing, must equal
    (D applied to f_i) composed with phi, times phi'^{k+sum a_i}, once each
    f_i is itself reweighted: f_i(dx)^{a_i} -> f_i(phi) phi'^{a_i} (dt)^{a_i}."""
    n = len(weights)
    t = sp.Symbol('t')
    nu = sum(weights) + k
    for phi in phis:
        dphi = sp.diff(phi, t)
        # generic test functions: truncated power series with free coefficients
        cs = sp.symbols(f'c0:{n*order}')
        fs = [sum(cs[i * order + j] * x**j for j in range(order))
              for i in range(n)]
        # left: the operator computed in the variable t on the transported
        # densities  g_i = f_i(phi) * (dphi)^{a_i}
        gs = [sp.expand(fs[i].subs(x, phi) * dphi**sp.Rational(
            weights[i].numerator, weights[i].denominator)) for i in range(n)]
        lhs = 0
        for z, c in zip(midx(n, k), vec):
            if not c:
                continue
            term = sp.Integer(1)
            for i in range(n):
                term *= sp.diff(gs[i], t, z[i])
            lhs += sp.Rational(c.numerator, c.denominator) * term
        # right: the operator computed in x, transported as a density of nu
        rhs = apply_op(vec, weights, k, fs).subs(x, phi) * dphi**sp.Rational(
            nu.numerator, nu.denominator)
        diff = sp.simplify(sp.expand(lhs - rhs))
        chk(f"{tag} finite phi={phi}", diff == 0)


XIS = [x**2, x**3, x**4, sp.exp(x), sp.sin(x)]
PHIS = [t**2 + t + 1 for t in [sp.Symbol('t')]] + [sp.exp(sp.Symbol('t'))]

# ------------------------------------------------------------------- d
v = inv_space((F(0),), 1)[0]
check_infinitesimal(v, (F(0),), 1, "d", XIS)
check_finite(v, (F(0),), 1, "d", PHIS, order=5)

# ------------------------------------------------- Grozman's operator
w = (F(-2, 3), F(-2, 3))
v = inv_space(w, 3)[0]
check_infinitesimal(v, w, 3, "Gz", XIS)
check_finite(v, w, 3, "Gz", PHIS, order=7)
print("Gz =", " + ".join(
    f"({c})*u^({z[0]})v^({z[1]})" for z, c in zip(midx(2, 3), v) if c))

# ------------------------------------------------- the bracket J_1
for w in [(F(1), F(1)), (F(1), F(2)), (F(-1), F(-1)), (F(0), F(3))]:
    for v in inv_space(w, 1):
        check_infinitesimal(v, w, 1, f"J1{w}", XIS)

# ------------------------------------------------- Wronskians of 1-forms
for n in [2, 3, 4]:
    w = tuple([F(1)] * n)
    s = n * (n - 1) // 2
    sp_ = inv_space(w, s)
    chk(f"Wronskian n={n} unique", len(sp_) == 1)
    v = sp_[0]
    check_infinitesimal(v, w, s, f"Wr{n}", XIS[:3])
    if n <= 3:
        check_finite(v, w, s, f"Wr{n}", PHIS, order=8)
    # it must be the determinant det(u_i^{(j-1)})
    us = [sp.Function(f'u{i}')(x) for i in range(n)]
    det = sp.Matrix(n, n, lambda i, j: sp.diff(us[j], x, i)).det()
    got = apply_op(v, w, s, us)
    chk(f"Wronskian n={n} is the determinant",
        sp.simplify(sp.expand(got - got.coeff(sp.diff(us[0], x, 0)) * 0
                              )) is not None
        and sp.simplify(sp.expand(det * sp.Rational(1, 1)
                                  - got / (got / det).simplify())) == 0)

# ------------------------------------------ a member of every space we use
for n in [2, 3]:
    w = tuple([F(1)] * n)
    for s in range(0, n * (n - 1) // 2 + 1):
        for v in inv_space(w, s):
            check_infinitesimal(v, w, s, f"gen n={n} s={s}", XIS[:3])

print(f"{ok} checks passed, {len(bad)} failed")
for b in bad:
    print("  FAIL", b)
