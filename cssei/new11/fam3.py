"""The three families T(N;g) = Span{H_f + f H_g | f != x^top} in vect(2;N), g = x_1, x_2, x^top:
closure, simplicity, centroid, restrictedness for a range of N (no invariant counts)."""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from xsearch import space
for p in ('..', '../rand', '../cmp'): sys.path.insert(0, os.path.join(HERE, p))
from gf2 import rref, rank
from lie import factors, centroid_dim, restricted
from rseries import table
res = {}
for N in [tuple(map(int, s.split(','))) for s in sys.argv[1:]]:
    monos, keys, br, H, fX, top = space(N)
    fs = [f for f in monos if f != top]
    for gname, g in (('x1', (1, 0)), ('x2', (0, 1)), ('top', top)):
        X = H(g)
        B = np.array([H(f) ^ fX(f, X) for f in fs], np.uint8)
        R, piv = rref(B.copy()); Bb = R[:len(piv)]
        try: ad = table(Bb, br)
        except AssertionError:
            print(N, gname, 'not closed', flush=True); continue
        d = ad.shape[0]; out = []; ab = factors(ad, np.random.default_rng(1), out)
        simp = ab == 0 and len(out) == 1 and out[0][0].shape[0] == d and out[0][1] == 'simple'
        info = dict(dim=d, simple=bool(simp), factors=[o[0].shape[0] for o in out], abelian=ab)
        if simp:
            info['centroid'] = centroid_dim(ad, out[0][2]); info['restricted'] = bool(restricted(ad))
        print(N, gname, info, flush=True); res['%s %s' % (N, gname)] = info
json.dump(res, open(os.path.join(HERE, 'fam3.json'), 'w'), indent=1)
