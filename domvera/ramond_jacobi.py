#!/usr/bin/env python3
"""Statement 8.3.2 of the text: the Jacobi identity of the k^M(1|7) model of nis.py on random triples."""
import sys, os, random
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from nis import kL
alg = kL(7, 3, ramond=True); bas = alg.basis; rnd = random.Random(7)
def br(X, Y):
    out = {}
    for x, a in X.items():
        for y, b in Y.items():
            r = alg.bracket(x, y)
            if r is None: return None
            for w, c in r.items(): out[w] = out.get(w, 0) + a * b * c
    return {k: v for k, v in out.items() if v}
p = lambda b: len(b[1]) % 2
tot = bad = 0
while tot < 3000:
    x, y, z = [rnd.choice(bas) for _ in range(3)]
    if abs(x[0]) + abs(y[0]) + abs(z[0]) > 3: continue
    X, Y, Z = {x: 1}, {y: 1}, {z: 1}
    yz, xy, xz = br(Y, Z), br(X, Y), br(X, Z)
    if None in (yz, xy, xz): continue
    l = br(X, yz); a = br(xy, Z); b = br(Y, xz)
    if None in (l, a, b): continue
    s = (-1) ** (p(x) * p(y)); d = dict(l)
    for k, v in a.items(): d[k] = d.get(k, 0) - v
    for k, v in b.items(): d[k] = d.get(k, 0) - s * v
    tot += 1; bad += any(d.values())
print("k^M(1|7) Jacobi: %d failures among %d triples" % (bad, tot))
