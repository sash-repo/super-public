"""Statement 5.A.8.3: is the Schouten-type form of an odd metric a contraction of the curvature?
Conformally flat test: g = lam^2 g0 (g0 constant odd), where any normal rho^0 must equal
-Hess_{g0}phi + dphi dphi  (mod c*g, c odd: the g_2 gauge).  Fit by the 32 contractions of oddexample."""
import itertools, sympy as sp
from oddexample import A, N, P, x, y, ev, evM, contractions
from oddgeom import geometry_odd
g0 = [[{} for _ in range(N)] for _ in range(N)]
g0[0][2] = g0[2][0] = {(): sp.Integer(1)}; g0[1][3] = g0[3][1] = {(): sp.Integer(1)}
lam = {(): 1 + x**2 + y, (0, 1): 1 + x}
g = [[A.mul(A.mul(lam, lam), g0[i][j]) for j in range(N)] for i in range(N)]
geo = geometry_odd(A, g)
C = contractions(geo, g)
for k in list(C):
    T = C[k]; C[('T',) + k] = [[A.sc((-1)**(P(a)*P(s)), T[s][a]) for s in range(N)] for a in range(N)]
C['Ric'] = evM(geo['Ric'])
from oddexample import rfull
Rf = rfull(geo, g); Gi = evM(geo['Gi'])
for tr, xs, ys in itertools.product((0, 1), repeat=3):   # contraction of the last pair: the 2-form str R
    T = [[{} for _ in range(N)] for _ in range(N)]
    for a, b in itertools.product(range(N), repeat=2):
        acc = {}
        for c, d in itertools.product(range(N), repeat=2):
            h = Gi[d][c] if tr else Gi[c][d]
            if h and Rf[a, b, c, d]: acc = A.add(acc, A.sc((-1)**(xs*P(c) + ys*P(d)), A.mul(h, Rf[a, b, c, d])))
        T[a][b] = acc
    C[('strR', tr, xs, ys)] = T
print("contractions in the fit:", len(C))
il = A.inv(lam); dphi = [A.mul(il, A.d(b, lam)) for b in range(N)]
H = evM([[A.d(b, dphi[d]) for d in range(N)] for b in range(N)])
dd = evM([[A.mul(dphi[b], dphi[d]) for d in range(N)] for b in range(N)])
gP = evM(g)
for sgnA in (0, 1):
    tgt = [[A.sc((-1)**(sgnA*P(a)), A.add(A.neg(H[a][s]), dd[a][s])) for s in range(N)] for a in range(N)]
    cs = sp.symbols('k0:%d' % len(C)); c0, c1 = sp.symbols('c0 c1')
    cg = {(0,): c0, (1,): c1}
    eqs = []
    keys = list(C)
    for a, s in itertools.product(range(N), repeat=2):
        expr = A.add(*[A.sc(cs[i], C[k][a][s]) for i, k in enumerate(keys)], A.mul(cg, gP[a][s]), A.neg(tgt[a][s]))
        eqs += list(expr.values())
    sol = sp.solve(eqs, list(cs) + [c0, c1], dict=True)
    print("convention (-1)^{p(a)}" if sgnA else "plain", "solvable:", bool(sol), sol[:1], flush=True)
