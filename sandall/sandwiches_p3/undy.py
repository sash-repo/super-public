"""Statement SSkrheight 1): dy(10;un), sdy(10;un) with the height of u8 (degree 4) raised: (ad d_8)^3 ?"""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); import dy3, unskry
from dpvect import DPVect
P = 3
LOG = os.path.join(HERE, "runs", "undy1766.log")
log = open(LOG, "w", buffering=1)
def say(m): print(m, flush=True); log.write(m + "\n")
dmax = int(sys.argv[1]); top = int(sys.argv[2]) if len(sys.argv) > 2 else 7
V0 = dy3.V
def comb(cs, Fs):
    out = {}
    for c, F in zip(cs, Fs):
        for k, v in F.items(): out[k] = (out.get(k, 0) + c*v) % P
    return {k: v for k, v in out.items() if v}
# the center of g_0 = gl(3): the combination of h1,h2,h3 commuting with x_i, y_i
H = [dy3.h1, dy3.h2, dy3.h3]; cen = None
for cs in itertools.product(range(3), repeat=3):
    if not any(cs): continue
    h = comb(cs, H)
    if all(not V0.bracket(h, e) for e in dy3.G0[:6]): cen = cs; break
say("center of g_0: %s . (h1,h2,h3)" % (cen,))
for kind in ("dy", "sdy"):
    h = [1]*10; h[top] = 2
    V = DPVect(dy3.DEG, dy3.WT, h, dy3.NAMES)
    g0 = list(dy3.G0)
    if kind == "sdy":
        # sl(3) = [gl(3), gl(3)]: x_i, y_i and h1=[x1,y1], h2=[x2,y2]
        g0 = dy3.G0[:8]
    layers = {}
    for D in dy3.GM:
        layers.setdefault(V0.fdeg(D), []).append(dict(D))
    layers[0] = [dict(D) for D in g0]
    unskry.prolong(V, layers, dmax, lambda m: None)
    dtop = {(tuple([0]*10), top): 1}
    nz = 0; ex = None
    for dg, fields in sorted(layers.items()):
        for D in fields:
            B = V.bracket(dtop, V.bracket(dtop, V.bracket(dtop, D)))
            if B:
                nz += 1
                if ex is None: ex = (dg, V.show(B))
    say("%s heights %s: dims %s; (ad d_u%d)^3 nonzero on %d fields; first %s"
        % (kind, h, {d: len(L) for d, L in sorted(layers.items())}, top+1, nz, ex))
