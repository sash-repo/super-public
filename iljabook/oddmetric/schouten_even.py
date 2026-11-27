"""Control for schouten.py: an EVEN metric g = lam^2 g0 on R^{3|2} (N = 3 - 2 = 1):
target rho^0 = -Hess phi + dphi dphi - 1/2|grad phi|^2 g must be fitted by Ric and S g."""
import itertools, sympy as sp
from stractor import Alg, geometry
x, y, z = sp.symbols('x y z')
A = Alg([x, y, z], 2); N = A.N; P = A.par
PT = {x: sp.Rational(3, 7), y: sp.Rational(-5, 11), z: sp.Rational(2, 13)}
ev = lambda f: A.clean({k: sp.sympify(v).subs(PT) for k, v in f.items()})
g0 = [[{} for _ in range(N)] for _ in range(N)]
for i in range(3): g0[i][i] = {(): sp.Integer(1)}
g0[3][4] = {(): sp.Integer(1)}; g0[4][3] = {(): sp.Integer(-1)}
lam = {(): 1 + x**2 + y + z, (0, 1): 1 + x}
g = [[A.mul(A.mul(lam, lam), g0[i][j]) for j in range(N)] for i in range(N)]
rs = lambda pa, pb, pc: (-1)**(pa*(1+pb+pc)); Ss = lambda pb: (-1)**pb
geo = geometry(A, g, rs, Ss)
il = A.inv(lam); dphi = [A.mul(il, A.d(b, lam)) for b in range(N)]
H = [[ev(A.d(b, dphi[d])) for d in range(N)] for b in range(N)]
dd = [[ev(A.mul(dphi[b], dphi[d])) for d in range(N)] for b in range(N)]
k1, k2, k3, k4 = sp.symbols('k1 k2 k3 k4')
eqs = []
for a, s in itertools.product(range(N), repeat=2):
    e = A.add(A.sc(k1, ev(geo['Ric'][a][s])), A.sc(k2, ev(A.mul(geo['S'], g[a][s]))), A.mul({(): k3, (0, 1): k4}, ev(g[a][s])),
              A.neg(A.add(A.neg(H[a][s]), dd[a][s])))
    eqs += list(e.values())
print("even control: fit -Hess+dphidphi = k1 Ric + k2 S g + k3 g:", sp.solve(eqs, [k1, k2, k3, k4], dict=True))
