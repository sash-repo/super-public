"""Targeted invariance test of B_s(f,g) = Res Int t^s f g on k^M(1|7) (zeta-realization, bracket (kb) of the text; (bf) is s=0):
for random f, g pick h dual to a term of [f,g] (so that B_s([f,g],h) != 0) and compare with B_s(f,[g,h])."""
import itertools, random, sympy as sp, sys
from kMperfect import setup
from volM import t, mul, par
n = 7
gens, br = setup(n); top = tuple(gens)
def coef_res(c, s):
    c = sp.expand(c * t ** s)
    return sp.Poly(c * t ** 30, t).coeff_monomial(t ** 29)
def B(f, g, s):
    c = mul(f, g).get(top, 0)
    return coef_res(c, s) if c != 0 else 0
R = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
mons = [{tuple(I): t ** k} for k in range(-2, 3) for d in range(n + 1) for I in itertools.combinations(gens, d)]
for s in (0, -1, 1):
    bad = tot = 0; ex = None
    for _ in range(150):
        f, g = R.choice(mons), R.choice(mons)
        fg = br(f, g)
        if not fg: continue
        m, v = R.choice(list(fg.items()))
        comp = tuple(x for x in gens if x not in m)
        k = sp.degree(sp.expand(v * t ** 10), t) - 10
        h = {comp: t ** (-1 - s - k)}
        a = B(fg, h, s); b = B(f, br(g, h), s)
        if a == 0 and b == 0: continue
        tot += 1
        if sp.simplify(a - b) != 0: bad += 1; ex = ex or (f, g, h, a, b)
    print("s=%d: invariance fails for %d of %d targeted triples; e.g. %s" % (s, bad, tot, ex), flush=True)
