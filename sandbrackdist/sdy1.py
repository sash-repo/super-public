import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); import dy3, unskry
from dpvect import DPVect
V0 = dy3.V
for name, g0 in (("G0[:8]", dy3.G0[:8]), ("G0", dy3.G0)):
    V = DPVect(dy3.DEG, dy3.WT, [1]*10, dy3.NAMES)
    layers = {}
    for D in dy3.GM: layers.setdefault(V0.fdeg(D), []).append(dict(D))
    layers[0] = [dict(D) for D in g0]
    unskry.prolong(V, layers, 3, lambda m: None)
    print(name, {d: len(L) for d, L in sorted(layers.items())})
# traces: is h3 in [g0,g0]?  print brackets of h's with x's
for nm, h in zip(("h1","h2","h3"), (dy3.h1, dy3.h2, dy3.h3)):
    print(nm, [V0.bracket(h, x) == x or V0.bracket(h,x)=={} for x in dy3.G0[:3]])
