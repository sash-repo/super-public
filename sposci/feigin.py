"""the duality chi -> chi_crit - chi of the superized Feigin theorem
preserves the zero locus of the Kac determinant (the shadow of item 3)),
level by level, for Virasoro, Neveu-Schwarz and Ramond, with
chi_crit = (1,26), (1/2,15), (5/8,15).  Exact (sympy)."""
import sympy as sp
b, r, s = sp.symbols('b r s')
Q2 = (b + 1/b)**2
Q2p = -(b - 1/b)**2           # b' = i b
cases = {
  # name: (c(Q2), h_{r,s}(Q2, x) with x = r b + s/b, level, h_crit, c_crit)
  'Vir': (lambda q: 1 + 6*q, lambda q, x: (q - x**2)/4, r*s, 1, 26),
  'NS':  (lambda q: sp.Rational(3, 2) + 3*q, lambda q, x: (q - x**2)/8, r*s/2,
          sp.Rational(1, 2), 15),
  'R':   (lambda q: sp.Rational(3, 2) + 3*q,
          lambda q, x: (q - x**2)/8 + sp.Rational(1, 16), r*s/2,
          sp.Rational(5, 8), 15),
}
n = 0
for name, (c, h, lev, hc, cc) in cases.items():
    # central charge: c_crit - c(b) = c(b')
    assert sp.simplify(cc - c(Q2) - c(Q2p)) == 0, name
    # h_crit - h_{r,s}(b) - level = h_{r,s}(b') with x' = i(r b - s/b)
    lhs = hc - h(Q2, r*b + s/b) - lev
    rhs = h(Q2p, sp.I*(r*b - s/b))
    assert sp.simplify(sp.expand(lhs - rhs)) == 0, name
    # and h_crit is the only value that works
    H = sp.Symbol('H')
    sol = sp.solve(sp.expand(H - h(Q2, r*b + s/b) - lev - rhs), H)
    assert sol == [hc], (name, sol)
    n += 3
    print(name, 'chi_crit =', (hc, cc), 'OK')
# Ramond: h_crit = c_crit/24, forced by chi(G_0)=0 and G_0^2 = L_0 - c/24
assert sp.Rational(15, 24) == sp.Rational(5, 8); n += 1
print(n, 'checks passed')
