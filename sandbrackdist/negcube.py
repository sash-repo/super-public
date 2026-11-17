#!/usr/bin/env python3
r"""negcube.py --- for Proposition Sknegcube: is D^3 = 0 for EVERY
D in g_- (not only for the basis), and do the Jacobson terms
s(x,y) = [x,[x,y]] + [y,[y,x]] vanish on g_{-1}?  Over F_3, all of g_- enumerated.
Also: the weights (mod 3) of the degree -3 fields of vect for dy(10)."""
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
D_ = HERE
sys.path.insert(0, D_)
from gvect import P
from skry3 import cube
from prolong import build
import dy3

def lin(V, fields, co):
    out = {}
    for c, D in zip(co, fields):
        if c:
            for k, v in D.items():
                out[k] = (out.get(k, 0) + c * v) % P
    return {k: v for k, v in out.items() if v}

def run(name, V, gm, gm1):
    n, nz = 0, 0
    for co in itertools.product(range(P), repeat=len(gm)):
        D = lin(V, gm, co)
        n += 1
        if cube(V, D):
            nz += 1
    js = 0
    for x, y in itertools.product(gm1, repeat=2):
        s = lin(V, [V.bracket(x, V.bracket(x, y)), V.bracket(y, V.bracket(y, x))], (1, 1))
        js += bool(s)
    print("%-4s: %d elements of g_-, nonzero cubes %d; nonzero Jacobson terms on g_-1 basis pairs %d"
          % (name, n, nz, js))

if __name__ == "__main__":
    for kind in ("my", "by"):
        V, gm, layers = build(kind)
        run(kind, V, gm, layers[-1])
    V = dy3.V
    run("dy", V, dy3.GM, dy3.GM[:3])
    # weights of the degree -3 fields of vect(10) for dy, and of the g_-1 basis
    print("dy wt of X1..X3:", [V.fwt(D) for D in dy3.GM[:3]])
    deg = dy3.DEG
    cands = []
    for a in range(10):
        for b in range(10):
            if deg[b] - deg[a] == 3:
                cands.append(("u%d d_u%d" % (a+1, b+1), tuple((dy3.WT[a][t]-dy3.WT[b][t]) % 3 for t in range(3))))
        if deg[a] == 3:
            cands.append(("d_u%d" % (a+1), tuple((-dy3.WT[a][t]) % 3 for t in range(3))))
    print("dy degree -3 fields, weights mod 3:", cands)
