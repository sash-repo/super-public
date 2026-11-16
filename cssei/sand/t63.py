"""T(N;g) for n1+n2=6 (dim 63): export tables; generic rank of ad x, compare with the two
63-dimensional constituents found by the random search (blocks/g2037408.npz, g2046526.npz)."""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); N11 = os.path.join(HERE, '..', 'new11')
sys.path.insert(0, N11)
for p in ('..', '../rand', '../cmp'): sys.path.insert(0, os.path.join(N11, p))
from xsearch import space
from gf2 import rref
from rseries import table
rng = np.random.default_rng(7)
def f2rank(M):
    R, piv = rref(M.copy() % 2); return len(piv)
def genrank(ad, trials=200):
    n = ad.shape[0]; best = 0
    for _ in range(trials):
        x = rng.integers(0, 2, n)
        A = np.tensordot(x, ad, axes=1) % 2
        best = max(best, f2rank(A.astype(np.uint8)))
    return best
out = {}
for N in [(3, 3), (2, 4), (4, 2)]:
    monos, keys, br, H, fX, top = space(N)
    fs = [f for f in monos if f != top]
    for gname, g in (('x1', (1, 0)), ('top', top)):
        X = H(g)
        B = np.array([H(f) ^ fX(f, X) for f in fs], np.uint8)
        R, piv = rref(B.copy()); Bb = R[:len(piv)]
        try:
            ad = table(Bb, br)
        except AssertionError:
            print(N, gname, 'not closed', flush=True); continue
        n = ad.shape[0]
        tab = [[int(sum(int(ad[i][k, j]) << k for k in range(n))) for j in range(n)] for i in range(n)]
        name = 'T_%d%d_%s' % (N[0], N[1], gname)
        json.dump({'name': name, 'dim': n, 'tab': tab}, open(os.path.join(HERE, name + '.json'), 'w'))
        r = genrank(ad)
        print(name, n, 'generic rank ad x', r, flush=True); out[name] = r
for s in ('2037408', '2046526'):
    ad = np.load(os.path.join(HERE, '..', 'blocks', 'g%s.npz' % s))['ad']
    print('constituent', s, ad.shape[0], 'generic rank ad x', genrank(ad), flush=True)
