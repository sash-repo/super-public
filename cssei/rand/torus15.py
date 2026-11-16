"""Exhaustive: the maximal rank of a torus (commuting toral elements, closed under +) in Der L15.
A torus of rank r has 2^r - 1 nonzero toral elements; extend tori one element at a time."""
import numpy as np, glob
from gf2 import mm
exec(open('der15.py').read().split('# toral elements')[0])
n = len(Dm); tor = []
cur = np.zeros((d, d), np.uint8)
for g in range(1, 1 << n):
    b = (g & -g).bit_length() - 1
    cur ^= Dm[b]
    if not (mm(cur, cur) ^ cur).any(): tor.append(cur.copy())
T = len(tor)
key = {t.tobytes(): i for i, t in enumerate(tor)}
C = np.zeros((T, T), bool)
for i in range(T):
    for j in range(i + 1, T):
        C[i, j] = C[j, i] = not (mm(tor[i], tor[j]) ^ mm(tor[j], tor[i])).any()
# tori as frozensets of indices of their nonzero elements
level = {frozenset([i]) for i in range(T)}
r = 1
while True:
    nxt = set()
    for S in level:
        cand = np.nonzero(C[list(S)].all(axis=0))[0]
        for c in cand:
            if c in S: continue
            new = set(S) | {c}
            for s in S:
                new.add(key[(tor[s] ^ tor[c]).tobytes()])
            nxt.add(frozenset(new))
    if not nxt: break
    level = nxt; r += 1
    print('tori of rank', r, ':', len(level), flush=True)
print('toral elements', T, '; maximal torus rank', r)
