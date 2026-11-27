"""Statement 5.A.7.4, item 1): Ricci of a generic odd metric on R^{2|2} (constant normal form + every quadratic
monomial with a random coefficient in every entry): anti-symmetric?  (exact)"""
import itertools, random, sympy as sp
from oddexample import A, N, P, x, y, is_sym
from oddgeom import geometry_odd
random.seed(1847)
mon = {0: [((), x**2), ((), x*y), ((), y**2), ((0, 1), sp.Integer(1))],      # even monomials of degree 2
       1: [((0,), x), ((0,), y), ((1,), x), ((1,), y)]}                       # odd ones
g = [[{} for _ in range(N)] for _ in range(N)]
for b in range(N):
    for c in range(b, N):
        par = (P(b) + P(c) + 1) % 2
        e = {}
        if par == 0 and P(b) != P(c) and c == b + 2: e[()] = sp.Integer(1)
        for mo, f in mon[par]:
            e[mo] = e.get(mo, 0) + sp.Rational(random.randint(-9, 9), random.randint(1, 5))*f
        e = A.clean(e)
        if b == c and P(b) == 1: e = {}          # odd-odd diagonal must vanish (anti-symmetric block)
        g[b][c] = e
        g[c][b] = e if (P(b)*P(c)) % 2 == 0 else A.neg(e)
geo = geometry_odd(A, g)
print("generic odd metric: Ric anti-symmetric:", is_sym(geo['Ric'], -1), " symmetric part zero; Ric nonzero:",
      any(e for r in geo['Ric'] for e in r))
