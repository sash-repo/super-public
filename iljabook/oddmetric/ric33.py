"""Statement 5.A.7.4, item 2), in superdimension (3|3); usage: python ric33.py cflat | generic | both.
1) generic odd metric (constant normal form + random quadratic terms): Ricci anti-symmetric?
2) conformally flat odd metric lam^2 g0: Ricci = 0?"""
import sys, itertools, random, sympy as sp
from stractor import Alg
from oddgeom import geometry_odd
x, y, z = sp.symbols('x y z')
A = Alg([x, y, z], 3); N = A.N; P = A.par
print("N =", N, "parities", [P(i) for i in range(N)], flush=True)
def is_sym(T, sgn=1):
    return all(A.add(T[b][d], A.sc(-sgn*(-1)**(P(b)*P(d)), T[d][b])) == {} for b in range(N) for d in range(N))
ev_ = [x, y, z]
mon = {0: [((), u*v) for u, v in itertools.combinations_with_replacement(ev_, 2)] + [((i, j), sp.Integer(1)) for i, j in itertools.combinations(range(3), 2)],
       1: [((i,), u) for i in range(3) for u in ev_]}
random.seed(23781)
def generic():
    g = [[{} for _ in range(N)] for _ in range(N)]
    for b in range(N):
        for c in range(b, N):
            par = (P(b) + P(c) + 1) % 2
            e = {}
            if par == 0 and P(b) != P(c) and c == b + 3: e[()] = sp.Integer(1)
            for mo, f in mon[par]:
                if random.random() < 0.5:
                    e[mo] = e.get(mo, 0) + sp.Rational(random.randint(-9, 9), random.randint(1, 5))*f
            e = A.clean(e)
            if b == c and P(b) == 1: e = {}
            g[b][c] = e
            g[c][b] = e if (P(b)*P(c)) % 2 == 0 else A.neg(e)
    return g
which = sys.argv[1] if len(sys.argv) > 1 else 'both'
if which in ('generic', 'both'):
    geo = geometry_odd(A, generic())
    print("(3|3) generic odd metric: Ric anti-symmetric:", is_sym(geo['Ric'], -1), " symmetric:", is_sym(geo['Ric'], 1),
          " Ric nonzero:", any(e for r in geo['Ric'] for e in r), flush=True)
if which in ('cflat', 'both'):
    g0 = [[{} for _ in range(N)] for _ in range(N)]
    for b in range(3): g0[b][b+3] = g0[b+3][b] = {(): sp.Integer(1)}
    lam = {(): 1 + x**2 + y - z, (0, 1): 1 + x, (1, 2): z}
    g = [[A.mul(A.mul(lam, lam), g0[i][j]) for j in range(N)] for i in range(N)]
    geo = geometry_odd(A, g)
    print("(3|3) conformally flat: Ric zero:", not any(e for r in geo['Ric'] for e in r), flush=True)
