"""CSS invariants of the small members of T(N;x1), T(N;top) and their match in CSS App. C."""
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from xsearch import space
for p in ('..', '../rand', '../cmp'): sys.path.insert(0, os.path.join(HERE, p))
from gf2 import rref
from rseries import table, dimder
from count import invariants
css = json.load(open(os.path.join(HERE, '..', 'rand', 'css_inv.json')))
out = {}
for N, gname in [((1,1),'top'),((1,2),'x1'),((1,2),'top'),((2,1),'top'),((1,3),'x1'),((1,3),'top'),((2,2),'x1'),((2,2),'top'),((1,4),'x1'),((1,4),'top'),((3,2),'x1'),((2,3),'x1'),((3,2),'top')]:
    monos, keys, br, H, fX, top = space(N)
    g = {'x1': (1, 0), 'top': top}[gname]
    fs = [f for f in monos if f != top]
    B = np.array([H(f) ^ fX(f, H(g)) for f in fs], np.uint8)
    R, piv = rref(B.copy()); ad = table(R[:len(piv)], br); d = ad.shape[0]
    tab = [[int(sum(int(ad[i][k, j]) << k for k in range(d))) for j in range(d)] for i in range(d)]
    inv = [int(x) for x in invariants(tab)]
    m = [k for k, v in css.items() if v == [d] + inv]
    dd = dimder(ad) if d <= 31 else None
    print(N, gname, d, inv, 'Der', dd, 'CSS:', m or '-', flush=True)
    out['%s %s' % (N, gname)] = dict(dim=d, inv=inv, dimDer=dd, css=m)
    json.dump({'name': 'T(%s;%s)' % (N, gname), 'dim': d, 'tab': tab}, open(os.path.join(HERE, 'T_%d%d_%s.json' % (N + (gname,))), 'w'))
json.dump(out, open(os.path.join(HERE, 'small.json'), 'w'), indent=1)
