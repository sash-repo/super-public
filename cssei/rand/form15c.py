"""Which 2-forms (alternating part a dx1dx2 + divided squares b_i dx_i^(2), coefficients in
O(2;(2,2)), the divided-power de Rham complex at p = 2) does the 15-dim algebra preserve?
Uses nalt2.py (Lie derivative on dx_i^(2): L_D dx_i^(2) = d(D_i) dx_i)."""
import sys, os
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from nalt2 import Alg, lie, g2_basis, flatten2, addform, closed_space, nondeg, gram, det
from exactlin import nullspace
exec(open('form15.py').read().split('# unknown g')[0].split('B = [')[0].replace('import numpy as np, itertools',''))
src = open('form15.py').read()
B = eval('[' + src.split('B = [')[1].split(']\n')[0] + ']', {'M': lambda e1, e2: e1 | (e2 << 2)})
alg = Alg(2, (2, 2))
def fieldof(D):
    v = [0, 0]
    for (m, i) in D:
        v[i] ^= 1 << alg.idx[(m & 3, m >> 2)]
    return v
Ds = [fieldof(D) for D in B]
b2 = g2_basis(2)
gens, cols = [], []
for k in b2:
    for t in range(alg.dim):
        om = {k: 1 << t}; gens.append(om)
        col = []
        for D in Ds:
            col += flatten2(alg, lie(alg, D, om), b2)
        cols.append(col)
M = [[cols[c][r] for c in range(len(cols))] for r in range(len(cols[0]))]
ns = nullspace(M, p=2)
def show(om):
    parts = []
    for k, c in om.items():
        f = ' + '.join(''.join(('x%d^(%d)' % (i+1, e) if e > 1 else 'x%d' % (i+1)) for i, e in enumerate(alg.mons[t]) if e) or '1'
                       for t in range(alg.dim) if c >> t & 1)
        parts.append('(%s) %s' % (f, 'dx1 dx2' if k[0] == 'a' else 'dx%d^(2)' % (k[1] + 1)))
    return ' + '.join(parts)
print('invariant 2-forms:', len(ns))
for v in ns:
    om = {}
    for c, coef in enumerate(v):
        if int(coef) % 2: addform(om, gens[c])
    print('  ', show(om), '  nondegenerate:', nondeg(alg, om))
