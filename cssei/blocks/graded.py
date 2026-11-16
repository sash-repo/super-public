#!/usr/bin/env python3
"""Regenerate a subalgebra of search2 by its seed and, if it is Z-graded (spanned by the
homogeneous components of its elements), write its table in a homogeneous basis with degrees.
usage: graded.py SEED OUT.npz"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rand'))
import search as S, search2 as S2
from lie import closure
from gf2 import Echelon, rref, mm

seed = int(sys.argv[1]); out = sys.argv[2]
S.AMS = S2.make()
rng = np.random.default_rng(seed)
a = S.AMS[seed % len(S.AMS)]
mode, gens = S2.gens_for(a, rng)
E = closure(a, gens, a.L)
print(a.name, mode, 'dim U', E.dim())
degs = sorted(set(a.deg.tolist()))
H = Echelon(a.L)
comps = []
for dg in degs:
    P = E.E.copy(); P[:, a.deg != dg] = 0
    R, piv = rref(P)
    if len(piv): comps.append((dg, R[:len(piv)]))
tot = sum(len(R) for _, R in comps)
print('graded' if tot == E.dim() else 'NOT graded (span of components %d)' % tot)
if tot != E.dim(): sys.exit(1)
B = np.vstack([R for _, R in comps]); deg = np.concatenate([[dg] * len(R) for dg, R in comps])
d = len(B)
# coordinates in basis B: B rows have disjoint degree supports, each block in rref
Rall, pivall = rref(B)
assert len(pivall) == d
ad = np.zeros((d, d, d), np.uint8)
for i in range(d):
    Y = a.br(B[i], B)          # rows j: [b_i, b_j]
    # solve Y = C B
    Rb, piv = rref(B)
    C = Y[:, piv]              # since B rows combine... use generic solve
    # generic solve: augment
    aug = np.hstack([B.T, Y.T]) % 2
    R2, p2 = rref(aug.astype(np.uint8))
    assert all(p < d for p in p2)
    X = np.zeros((d, d), np.uint8)
    for r, p in enumerate(p2): X[p] = R2[r, d:]
    assert not (mm(X.T, B) ^ Y).any()
    ad[i] = X            # column j = coords of [b_i,b_j]
np.savez_compressed(out, ad=ad, deg=deg)
print('degrees', {int(k): int((deg == k).sum()) for k in sorted(set(deg.tolist()))})
