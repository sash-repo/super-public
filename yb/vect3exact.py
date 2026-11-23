"""Claim 2.3, a cross-check over Q: on vect(0|3), with l = (xi_1 d_2)^* + (xi_2 d_3)^* +
(xi_1 xi_2 xi_3 d_1)^* and B(x,y) = l([x,y]), the matrix B is invertible, r = B^{-1} is
super-anti-symmetric, and r satisfies the CYBE exactly (every component of the left-hand side
vanishes).  The structure constants are those of cocyc.py, lifted from F_P to the integers
(they are 0, +-1, +-2).  Run:  python3 vect3exact.py
"""
import sys
from fractions import Fraction as F
import os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"), os.path.dirname(os.path.abspath(__file__))]
import cocyc as C, exactlin
n = 3; vb, idx = C.build(n, 'vect'); N = len(vb); par = [C.vf_par(X) for X in vb]
def sgn(v): return v - C.P if v > C.P // 2 else v
S = {(i, j): {k: sgn(v) for k, v in C.coords(C.vf_br(vb[i], vb[j]), idx).items() if v} for i in range(N) for j in range(N)}
l = {idx[((0,), 1)]: 1, idx[((1,), 2)]: 1, idx[((0, 1, 2), 0)]: 1}
B = [[F(sum(l.get(k, 0) * c for k, c in S[(i, j)].items())) for j in range(N)] for i in range(N)]
assert exactlin.rank(B) == N
# inverse over Q via solve
I = [[F(int(i == j)) for j in range(N)] for i in range(N)]
cols = [exactlin.solve(B, [I[i][j] for i in range(N)]) for j in range(N)]
Bi = [[F(cols[j][i]) for j in range(N)] for i in range(N)]
assert all(sum(B[i][k] * Bi[k][j] for k in range(N)) == (i == j) for i in range(N) for j in range(N))
r = Bi
skew = all(r[b][a] * (-1 if (par[a] and par[b]) else 1) + r[a][b] == 0 for a in range(N) for b in range(N))
out = {}
nz = [(a, b, r[a][b]) for a in range(N) for b in range(N) if r[a][b]]
for a, b, rab in nz:
    for c, d, rcd in nz:
        f = rab * rcd
        for k, v in S[(a, c)].items(): out[(k, b, d)] = out.get((k, b, d), 0) + (-1 if par[a] and par[c] else 1) * f * v
        for k, v in S[(b, c)].items(): out[(a, k, d)] = out.get((a, k, d), 0) + f * v
        for k, v in S[(b, d)].items(): out[(a, c, k)] = out.get((a, c, k), 0) + (-1 if par[b] and par[c] else 1) * f * v
print('basis', N, 'nonzero entries of r:', len(nz), 'super-anti-symmetric:', skew,
      'nonzero CYBE components over Q:', sum(1 for v in out.values() if v), 'of', len(out))
