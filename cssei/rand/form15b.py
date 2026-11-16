"""Same as form15.py, g now in O(2;(H1,H2)) with larger heights (divided powers)."""
import numpy as np, sys
from gf2 import nullspace
exec(open('form15.py').read().split('# unknown g')[0].replace('Ns = (2, 2); off = (0, 2); N = 16', 'Ns=(2,2); off=(0,2); N=16'))
H1, H2 = int(sys.argv[1]), int(sys.argv[2])
def conv(m):  # packed (2,2) -> packed (H1,H2)
    return (m & 3) | ((m >> 2) << H1)
NN = 1 << (H1 + H2); o2 = H1
def f2(a, i): return (a & ((1 << H1) - 1)) if i == 0 else a >> o2
def mulb(a, b): return (a | b) if a & b == 0 else None
def db(a, i):
    if f2(a, i) == 0: return None
    return a - (1 if i == 0 else 1 << o2)
rows = []
for D in B:
    eq = np.zeros((NN, NN), np.uint8)
    for (m, i) in D:
        mm_ = conv(m)
        for k in range(NN):
            p = mulb(k, mm_)
            if p is None: continue
            q = db(p, i)
            if q is None: continue
            eq[q, k] ^= 1
    rows.extend(eq)
Ker = nullspace(np.array(rows, np.uint8))
def name(k):
    e1, e2 = k & ((1 << H1) - 1), k >> o2
    s = ''.join(('x%d^(%d)' % (i+1, e) if e > 1 else 'x%d' % (i+1)) for i, e in ((0, e1), (1, e2)) if e)
    return s or '1'
for v in Ker: print(' + '.join(name(k) for k in np.nonzero(v)[0]))
print('solutions', len(Ker))
