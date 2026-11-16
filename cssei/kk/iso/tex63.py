"""Export the dim-63 members T(N;g), N=(3,3),(2,4),(4,2), g=x1,top as
json tables tab[i][j] = bitmask of [e_i,e_j], for comparison with P_2(6)."""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(HERE, '..', '..', 'new11'); sys.path.insert(0, NEW)
for p in ('..', '../rand', '../cmp'): sys.path.insert(0, os.path.join(NEW, p))
from xsearch import space
from gf2 import rref
from rseries import table
for N in [(3, 3), (2, 4), (4, 2)]:
    monos, keys, br, H, fX, top = space(N)
    fs = [f for f in monos if f != top]
    for gname, g in (('x1', (1, 0)), ('top', top)):
        X = H(g)
        B = np.array([H(f) ^ fX(f, X) for f in fs], np.uint8)
        R, piv = rref(B.copy()); Bb = R[:len(piv)]
        ad = table(Bb, br); n = ad.shape[0]
        tab = [[int(sum(int(ad[i][k, j]) << k for k in range(n))) for j in range(n)] for i in range(n)]
        name = 'T_%d%d_%s' % (N + (gname,))
        json.dump({'name': name, 'dim': n, 'tab': tab}, open(os.path.join(HERE, name + '.json'), 'w'))
        print(name, n, flush=True)
