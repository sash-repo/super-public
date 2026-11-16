"""Numerical check of Lemma 7.3 (the defect of B_ij under K_{t theta}) over F_p, for all a,b,c in F_p, i,j < 2P.
Usage: python3 bindef.py p N"""
import sys
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from binnum import basics, bilin
from osp1 import generators, rep
from binpred import al
p, N = int(sys.argv[1]), int(sys.argv[2])
S, D, pows, prod = basics(p, N); n, P = S['n'], S['P']
g = generators(S, -1)
def defect_num(T, a, b, c, pi):
    X = {lam: rep(S, g, lam)[4][0] for lam in (a, b, c)}   # L^lam_{K_{t theta}}
    out = {}
    def addto(k, w):
        out[k] = (out.get(k, 0) + w) % p
    s = -1 if pi else 1
    for (z, u, v), w in T.items():
        for y in range(n):                      # L^c T
            if X[c][y][z]: addto((y, u, v), X[c][y][z]*w)
    for (z, u, v), w in T.items():
        for u2 in range(n):                     # T(L^a e_u2, e_v): coefficient X[a][u][u2]
            if X[a][u][u2]: addto((z, u2, v), -s*X[a][u][u2]*w)
        for v2 in range(n):
            if X[b][v][v2]: addto((z, u, v2), -s*(-1 if (u % 2) else 1)*X[b][v][v2]*w)
    return {k: w for k, w in out.items() if w}
bad = 0; tot = 0
for a in range(p):
  for b in range(p):
    for c in range(p):
      for i in range(2*P):
        for j in range(2*P):
          tot += 1
          T = bilin(S, pows, prod, 0, i, j)
          lhs = defect_num(T, a, b, c, (i+j) % 2)
          rhs = {}
          def acc(Tm, f):
              for k, w in Tm.items(): rhs[k] = (rhs.get(k, 0) + f*w) % p
          acc(bilin(S, pows, prod, 1, i, j), (c-a-b-i-j))
          if i: acc(bilin(S, pows, prod, 0, i-1, j), al(i, a, p))
          if j: acc(bilin(S, pows, prod, 0, i, j-1), (-1)**i*al(j, b, p))
          rhs = {k: w for k, w in rhs.items() if w}
          if lhs != rhs: bad += 1; 
          if lhs != rhs and bad < 5: print("bad", a, b, c, i, j)
print(tot, "checks,", bad, "bad")
