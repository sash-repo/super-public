"""Same as kM7form.py, s=0 (the form (bf)), split by the parity of f: is (bf) at least g_ev-invariant?"""
import itertools, random, sympy as sp
from kM7form import B, br, mons, gens, R, t
from volM import par
for want in (0, 1):
    bad = tot = 0; ex = None
    pool = [m for m in mons if par(m) == want]
    for _ in range(200):
        f, g = R.choice(pool), R.choice(mons)
        fg = br(f, g)
        if not fg: continue
        m, v = R.choice(list(fg.items()))
        comp = tuple(x for x in gens if x not in m)
        k = sp.degree(sp.expand(v * t ** 10), t) - 10
        h = {comp: t ** (-1 - k)}
        a = B(fg, h, 0); b = B(f, br(g, h), 0)
        if a == 0 and b == 0: continue
        tot += 1
        if sp.simplify(a - b) != 0: bad += 1; ex = ex or (f, g, h, a, b)
    print("f %s: (bf) invariance fails for %d of %d; e.g. %s" % ("even" if want == 0 else "odd", bad, tot, ex), flush=True)
