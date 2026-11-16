"""Find g in O(2;(2,2)) with L_D(g dx1 dx2) = 0, i.e. d1(g D1) + d2(g D2) = 0, for every
D of the 15-dim algebra (seed 1001651 of runs.jsonl); also test forms outside O: none needed if found."""
import numpy as np, itertools
Ns = (2, 2); off = (0, 2); N = 16
def fld(a, i): return (a >> off[i]) & 3
def mul(a, b): return (a | b) if a & b == 0 else None
def d(a, i): return None if fld(a, i) == 0 else a - (1 << off[i])
# the basis as dicts {(mono, i)}; mono packed x1 in bits 0-1, x2 in bits 2-3
def M(e1, e2): return e1 | (e2 << 2)
B = [
 {(M(0,0),0), (M(3,1),1)}, {(M(0,0),1)}, {(M(1,0),0), (M(0,1),1)}, {(M(1,0),1)},
 {(M(2,0),0), (M(1,1),1)}, {(M(2,0),1)}, {(M(3,0),0), (M(2,1),1)}, {(M(3,0),1)},
 {(M(0,1),0), (M(3,2),1)}, {(M(1,1),0), (M(0,2),1)}, {(M(2,1),0), (M(1,2),1)}, {(M(3,1),0), (M(2,2),1)},
 {(M(0,2),0), (M(3,3),1)}, {(M(1,2),0), (M(0,3),1)}, {(M(2,2),0), (M(1,3),1)}]
# unknown g = sum c_k x^(k); condition: for each D, sum_i d_i(g D_i) = 0 -> linear in c
rows = []
for D in B:
    eq = np.zeros((N, N), np.uint8)   # eq[monomial of result][k]
    for (m, i) in D:
        for k in range(N):
            p = mul(k, m)
            if p is None: continue
            q = d(p, i)
            if q is None: continue
            eq[q, k] ^= 1
    rows.extend(eq)
import sys; sys.path.insert(0, '.')
from gf2 import nullspace
Ker = nullspace(np.array(rows, np.uint8))
def name(k):
    e1, e2 = k & 3, k >> 2
    s = ''.join(('x%d^(%d)' % (i+1, e) if e > 1 else 'x%d' % (i+1)) for i, e in ((0, e1), (1, e2)) if e)
    return s or '1'
for v in Ker:
    print(' + '.join(name(k) for k in np.nonzero(v)[0]))
print('solutions:', len(Ker))
