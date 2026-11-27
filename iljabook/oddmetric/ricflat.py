"""Statement 5.A.7.4, item 2), in superdimension (2|2): the Ricci tensor of a conformally flat odd
metric lam^2 g0 vanishes; the supertrace S of the explicit metric of oddexample.py."""
import sympy as sp
from oddexample import A, N, P, x, y, ev, is_sym
from oddgeom import geometry_odd, r_of
g0 = [[{} for _ in range(N)] for _ in range(N)]
g0[0][2] = g0[2][0] = {(): sp.Integer(1)}; g0[1][3] = g0[3][1] = {(): sp.Integer(1)}
lam = {(): 1 + x**2 + y, (0, 1): 1 + x}
for name, g in (("conformally flat", [[A.mul(A.mul(lam, lam), g0[i][j]) for j in range(N)] for i in range(N)]),):
    geo = geometry_odd(A, g)
    r = r_of(A, geo['Ric'], geo['Gi'])
    S1 = A.add(*[A.sc((-1)**P(a), r[a][a]) for a in range(N)]); S2 = A.add(*[r[a][a] for a in range(N)])
    print(name, "Ric anti:", is_sym(geo['Ric'], -1), "sym:", is_sym(geo['Ric'], 1), "Ric nonzero:", any(e for row in geo['Ric'] for e in row), "S (str):", S1, "S (tr):", S2)
from oddexample import metric
g = metric(); geo = geometry_odd(A, g); r = r_of(A, geo['Ric'], geo['Gi'])
print("random odd metric: S (str):", A.add(*[A.sc((-1)**P(a), r[a][a]) for a in range(N)]), " S (tr):", A.add(*[r[a][a] for a in range(N)]))
