"""Subsec. 8.2 a) of the paper: the 3x3 matrix with 2 on the main diagonal and 1 off it,
all generators even: dims of n_+ by degree up to 8, as the rank of the iterated-lowering
matrix Phi (exact, via exactlin).  The free Lie algebra on 3 generators has the same dims."""
import sys, time
from fractions import Fraction as F
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE, os.path.join(_HERE, "..")]
import sympy as sp
from exactlin import rank, backend
from shapovalov import free_lie_superalgebra, phi, perms_of
MAXH = int(sys.argv[1]) if len(sys.argv) > 1 else 8
vals = ['sym']
par = [0, 0, 0]
L = free_lie_superalgebra(3, MAXH, par=par)
out = open(os.path.join(_HERE, 'free111.log'), 'w')
def P(s):
    print(s, flush=True); out.write(s + '\n'); out.flush()
P('backend %s; free dims %s' % (backend(), {k: sum(len(v) for v in L[k].values()) for k in sorted(L)}))
for av in vals:
    A = [[sp.Integer(2),sp.Integer(1),sp.Integer(1)],[sp.Integer(1),sp.Integer(2),sp.Integer(1)],[sp.Integer(1),sp.Integer(1),sp.Integer(2)]]
    res = {}
    for k in range(1, MAXH + 1):
        t0 = time.time(); tot = 0
        for alpha in sorted(L.get(k, {})):
            basis = L[k][alpha]
            if k == 1:
                tot += len(basis); continue
            words = perms_of(alpha, 3)
            rows = []
            for u in basis:
                p = phi(u, A, 3, alpha, par)
                rows.append([F(int(sp.Rational(sp.expand(p.get(w, 0))).p), int(sp.Rational(sp.expand(p.get(w, 0))).q)) for w in words])
            tot += rank(rows)
        res[k] = tot
        P('a=%s deg %d dim %d [%.0fs]' % (av, k, tot, time.time() - t0))
    P('a=%s: %s' % (av, res))
