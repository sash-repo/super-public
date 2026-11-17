#!/usr/bin/env python3
"""Graded isomorphism gs(2) -> J_0 (degree-0 part of the core of BGLLS's sb~(7;1)) matching the root
decompositions: root vectors to root vectors (over F_2 with coefficient 1), the torus by any of the 6
invertible 2x2 matrices; every candidate checked on all 196 brackets.
In the text: proof of Theorem 4.6, 1) (J_0 of sb~(7;1) is isomorphic to gs(2))."""
import json, itertools, numpy as np
from fractions import Fraction as Fr
from lalg import CL, NAMES, NG, WTS
n = NG
Jw = {7: (0, -1, 1), 8: (-1, 0, 1), 9: (1, 1, -2), 10: (-1, -2, 3), 11: (-2, -1, 3), 12: (0, 0, 0), 13: (0, 0, 0),
      14: (-1, -1, 2), 15: (1, 0, -1), 16: (1, -1, 0), 17: (0, 1, -1), 18: (2, 1, -3), 19: (1, 2, -3), 20: (-1, 1, 0)}
B = json.load(open('sbt7_J0.json'))['tab']          # J_0 basis = J indices 7..20 in order
jidx = list(range(7, 21))
A = [[sum(int(CL[i, j, k]) << k for k in range(n)) for j in range(n)] for i in range(n)]
def brv(tab, a, b):
    out = 0; i = 0
    while a:
        if a & 1:
            bb, j = b, 0
            while bb:
                if bb & 1: out ^= tab[i][j]
                bb >>= 1; j += 1
        a >>= 1; i += 1
    return out
groots = {i: tuple(int(x) for x in WTS[i]) for i in range(n) if tuple(WTS[i]) != (0, 0)}
gzero = [i for i in range(n) if tuple(WTS[i]) == (0, 0)]
jroots = {t: Jw[j][:2] for t, j in enumerate(jidx) if Jw[j] != (0, 0, 0)}
jzero = [t for t, j in enumerate(jidx) if Jw[j] == (0, 0, 0)]
gl = list(groots.items()); r1, r2 = gl[0], None
for it in gl[1:]:
    if it[1][0] * r1[1][1] - it[1][1] * r1[1][0] != 0: r2 = it; break
found = []
for t1, t2 in itertools.permutations(jroots, 2):
    # linear map Q with Q r1 = jroots[t1], Q r2 = jroots[t2]
    a, b = r1[1], r2[1]; c, d = jroots[t1], jroots[t2]
    det = Fr(a[0] * b[1] - a[1] * b[0])
    # Q = [c d] [a b]^{-1}
    inv = [[Fr(b[1]) / det, Fr(-b[0]) / det], [Fr(-a[1]) / det, Fr(a[0]) / det]]
    Q = [[c[0] * inv[0][0] + d[0] * inv[1][0], c[0] * inv[0][1] + d[0] * inv[1][1]],
         [c[1] * inv[0][0] + d[1] * inv[1][0], c[1] * inv[0][1] + d[1] * inv[1][1]]]
    img = {}
    ok = True
    for i, w in groots.items():
        v = (Q[0][0] * w[0] + Q[0][1] * w[1], Q[1][0] * w[0] + Q[1][1] * w[1])
        hit = [t for t, u in jroots.items() if u == v]
        if len(hit) != 1: ok = False; break
        img[i] = hit[0]
    if not ok or len(set(img.values())) != 12: continue
    for Mh in [(1, 0, 0, 1), (0, 1, 1, 0), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 0)]:
        phi = {i: 1 << img[i] for i in groots}
        phi[gzero[0]] = (Mh[0] << jzero[0]) | (Mh[1] << jzero[1])
        phi[gzero[1]] = (Mh[2] << jzero[0]) | (Mh[3] << jzero[1])
        def ph(v):
            out = 0
            for i in range(n):
                if (v >> i) & 1: out ^= phi[i]
            return out
        if all(ph(A[i][j]) == brv(B, phi[i], phi[j]) for i in range(n) for j in range(n)):
            found.append((dict(img), Mh))
print('graded isomorphisms gs(2) -> J_0 found:', len(found))
if found:
    img, Mh = found[0]
    print('e.g.', {NAMES[i]: jidx[t] for i, t in img.items()}, 'torus', Mh)
