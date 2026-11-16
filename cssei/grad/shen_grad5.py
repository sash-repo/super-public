#!/usr/bin/env python3
"""The perfect g_0 (outer-type grading of gs(2)): its 3-dimensional ideals and their type."""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shen_grad3.py')).read().split("done = set()")[0])
for D in res:
    g0 = nullspace(D)
    if derived_series(g0) != [6]:
        continue
    ideals = set()
    for x in itertools.product([0, 1], repeat=6):
        if not any(x):
            continue
        v = sum(c * b for c, b in zip(x, g0)) % 2
        S = span_basis([v]); fr = [v]
        while fr:
            nf = []
            for w in fr:
                for a in g0:
                    u = br(a, w); k = len(S); S = span_basis(S + [u])
                    if len(S) > k:
                        nf.append(u)
            fr = nf
        if len(S) == 3:
            ideals.add(frozenset(vec2int(s) for s in S) if False else tuple(sorted(vec2int(w) for w in [sum(c*b for c,b in zip(y,S))%2 for y in itertools.product([0,1],repeat=3) if any(y)])))
    print('number of 3-dim ideals of g_0:', len(ideals))
    for I in ideals:
        B = [np.array([(v >> k) & 1 for k in range(n)], dtype=np.uint8) for v in I][:0]
        vs = [np.array([(v >> k) & 1 for k in range(n)], dtype=np.uint8) for v in I]
        Bs = span_basis(vs)
        print('  ideal: derived dim', len(span_basis([br(a, b) for a in Bs for b in Bs])),
              '; elements with (ad x)^2 = 0 on the ideal:',
              sum(1 for y in itertools.product([0,1],repeat=3) if any(y) and all(not br(sum(c*b for c,b in zip(y,Bs))%2, br(sum(c*b for c,b in zip(y,Bs))%2, z)).any() for z in Bs)))
    break
