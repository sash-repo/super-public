#!/usr/bin/env python3
"""Where the idempotent derivations D of gs(2) lie: ad g (14), the 2-envelope (17), Der g (20);
and dim of the torus they span (maximal tori of Der g containing them)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shen_grad.py')).read().split("inner = BitEchelon()")[0])
env = BitEchelon(); queue = []
for A in ADS:
    if env.add(flat(A)):
        queue.append(A.astype(np.int64))
while queue:
    A = queue.pop()
    S = (A @ A) % 2
    if env.add(flat(S.astype(np.uint8))):
        queue.append(S)
print('dim 2-envelope of ad g in End g:', len(env))
adsp = BitEchelon()
for A in ADS:
    adsp.add(flat(A))
from collections import Counter
cnt = Counter()
for D in res:
    f = flat(D)
    where = 'ad g' if adsp.reduce(f) == 0 else ('2-envelope' if env.reduce(f) == 0 else 'outside the 2-envelope')
    g1 = len(colspace(D))
    cnt[(where, 14 - g1, g1)] += 1
for k, v in sorted(cnt.items()):
    print(k, v)
# the toral elements of Der g: they span tori; rank of the F_2-span of all toral D
T = BitEchelon()
for D in res:
    T.add(flat(D))
print('F_2-span of all idempotent D: dim', len(T))
# commuting families: maximal set of pairwise commuting idempotents containing a given one
def comm(A, B):
    A = A.astype(np.int64); B = B.astype(np.int64)
    return not ((A @ B - B @ A) % 2).any()
best = 0
for D in res[:40]:
    fam = [D]
    for E in res:
        if all(comm(E, F) for F in fam):
            fam.append(E)
    S = BitEchelon()
    for F in fam:
        S.add(flat(F))
    best = max(best, len(S))
print('rank of a torus spanned by pairwise commuting idempotents (greedy, from 40 starts):', best)
