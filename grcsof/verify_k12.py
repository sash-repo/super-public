#!/usr/bin/env python3
"""Independent verification of classify_k12.py.

For random rational weights (lam_1,nu_1,lam_2,nu_2) we recompute the space of
invariant binary operators using ALL EIGHT generators of i(osp(2|2)) and a
much larger set of test monomials than the classification used, and we check
the explicit closed formulas found for orders 1, 2, 3.
"""
import sys
from sympy import symbols, S, Rational, Matrix, expand, simplify
from invar import gen_binary, eqs_binary, matrix_of
from k12 import D_xi, D_eta, d_t

l1, n1, l2, n2 = symbols('lambda_1 nu_1 lambda_2 nu_2')
WEIGHTS = [(S(0), S(0), S(0), S(0)),
           (Rational(1, 2), Rational(-1, 3), Rational(5, 2), Rational(2, 3)),
           (S(3), S(1), S(-2), S(4)),
           (Rational(-7, 3), Rational(5, 7), Rational(11, 5), Rational(-1, 2)),
           (S(1), S(0), S(1), S(0)),
           (S(2), S(-2), S(-3), S(3))]

print("A. dimension of the solution space, all 8 generators, nmax = order+4")
print("   (compare with classify_k12.py, which used only K_txi, K_teta)")
for s in range(-1, -5, -1):
    for w in (-2, -1, 0, 1, 2):
        terms = gen_binary(s, w)
        if not terms:
            continue
        dims = []
        for (a, b, c, d) in WEIGHTS:
            cs, eqs = eqs_binary(terms, a, b, c, d, a + c - s, b + d - w, w % 2,
                                 nmax=(-s) // 2 + 4)
            dims.append(len(matrix_of(eqs, cs).nullspace()))
        print(f"   s={s:3d} w={w:+d}: dims {dims}")
        sys.stdout.flush()

print()
print("B. the closed formulas (all 8 generators, nmax = 8)")
idop = lambda u: u
dxdy = lambda u: D_xi(D_eta(u))


def T(nm, P, Q, pQ):
    return (nm, P, Q, None, pQ)


def check(name, s, w, terms, coeffs, nmax=8):
    bad = []
    for (a, b, c, d) in WEIGHTS:
        cs, eqs = eqs_binary(terms, a, b, c, d, a + c - s, b + d - w, w % 2, nmax=nmax)
        sub = {cs[i]: coeffs[i].subs({l1: a, n1: b, l2: c, n2: d}) for i in range(len(cs))}
        res = [expand(e.subs(sub)) for e in eqs]
        if any(r != 0 for r in res):
            bad.append((a, b, c, d))
    print(f"   {'OK  ' if not bad else 'FAIL'}  {name}"
          + ('' if not bad else f"  at {bad}"))
    sys.stdout.flush()


# order 1, charge +1:  (lam_2+nu_2) D_xi(phi) psi - (-1)^p(phi) (lam_1+nu_1) phi D_xi(psi)
check("J^{1/2}_xi = (l2+n2) Dx(f)g - (-1)^p (l1+n1) f Dx(g)", -1, +1,
      [T('Dx x 1', D_xi, idop, 0), T('1 x Dx', idop, D_xi, 1)],
      [l2 + n2, -(l1 + n1)])
# order 1, charge -1
check("J^{1/2}_eta = (l2-n2) De(f)g - (-1)^p (l1-n1) f De(g)", -1, -1,
      [T('De x 1', D_eta, idop, 0), T('1 x De', idop, D_eta, 1)],
      [l2 - n2, -(l1 - n1)])

# order 2, charge 0: the two basis vectors printed by classify_k12.py,
# rewritten with the denominator (lam_2-nu_2) cleared.
terms2 = [T('1 x d', idop, d_t, 0), T('d x 1', d_t, idop, 0),
          T('1 x Dxe', idop, dxdy, 0), T('Dx x De', D_xi, D_eta, 1),
          T('De x Dx', D_eta, D_xi, 1), T('Dxe x 1', dxdy, idop, 0)]
v1 = [(-(l1 * (l2 + n2 + 1) - n1)) * (l2 - n2),
      (l2 + 1) * (l2 + n2) * (l2 - n2),
      -(l1 * n2 + l2 * n1),
      (l2 + 1) * (l2 + n2),
      (l2 + 1) * (l2 - n2),
      S(0)]
v2 = [(l1 + 1) * (l1 + n1) * (l2 - n2),
      -(l2 + 1) * (l1 + n1 + 2) * (l2 - n2),
      (l1 + 1) * (l1 + n1),
      -2 * (l1 + 1) * (l2 + 1),
      S(0),
      (l2 + 1) * (l2 - n2)]
check("order 2, basis vector 1", -2, 0, terms2, v1)
check("order 2, basis vector 2", -2, 0, terms2, v2)

# order 3, charge +1
terms3 = [T('1 x dDx', idop, lambda u: D_xi(d_t(u)), 1),
          T('d x Dx', d_t, D_xi, 1),
          T('Dx x d', D_xi, d_t, 0),
          T('dDx x 1', lambda u: D_xi(d_t(u)), idop, 0),
          T('Dx x Dxe', D_xi, dxdy, 0),
          T('Dxe x Dx', dxdy, D_xi, 1)]
v3 = [(l1 + 1) * (l1 + n1), -(l2 + 1) * (l1 + n1 + 2), -(l1 + 1) * (l2 + n2 + 2),
      (l2 + 1) * (l2 + n2), (l1 + 1), (l2 + 1)]
check("order 3, charge +1", -3, +1, terms3, v3)

terms3m = [T('1 x dDe', idop, lambda u: D_eta(d_t(u)), 1),
           T('d x De', d_t, D_eta, 1),
           T('De x d', D_eta, d_t, 0),
           T('dDe x 1', lambda u: D_eta(d_t(u)), idop, 0),
           T('De x Dxe', D_eta, dxdy, 0),
           T('Dxe x De', dxdy, D_eta, 1)]
v3m = [-(l1 + 1) * (l1 - n1), (l1 - n1) * (l2 + 1), (l1 + 1) * (l2 - n2),
       -(l2 + 1) * (l2 - n2), (l1 + 1), (l2 + 1)]
check("order 3, charge -1", -3, -1, terms3m, v3m)
