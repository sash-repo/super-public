"""Der(L) of the 15-dim algebra; toral elements of Der (D^2 = D); is L thin
(a torus T <= Der of rank 4 with 15 one-dimensional nonzero root spaces)?"""
import numpy as np, glob, itertools
from gf2 import mm, nullspace, rank, rref
z = [np.load(p) for p in glob.glob('found/*.npz') if str(np.load(p)['fp']).startswith('15:1:False:simple:14')][0]
ad = z['ad']; d = ad.shape[0]
def br(x, y):
    return mm((ad[np.nonzero(x)[0]].sum(0) & 1).astype(np.uint8), y[:, None])[:, 0]
# D e_j = sum_k D[k,j] e_k ; unknown index k*d+j ; D[e_i,e_j] = [De_i,e_j] + [e_i,De_j]
rows = []
for i in range(d):
    for j in range(i + 1, d):
        c = ad[i][:, j]
        for r in range(d):
            row = np.zeros(d * d, np.uint8)
            for k in np.nonzero(c)[0]: row[r * d + k] ^= 1          # (D c)_r
            for k in range(d):
                if ad[k][r, j]: row[k * d + i] ^= 1                  # [D e_i, e_j]_r = sum_k D[k,i] ad[k][r,j]
                if ad[i][r, k]: row[k * d + j] ^= 1                  # [e_i, D e_j]_r
            if row.any(): rows.append(row)
Der = nullspace(np.array(rows, np.uint8))
print('dim Der', len(Der), ' (dim L =', d, ', outer', len(Der) - d, ')')
Dm = [v.reshape(d, d) for v in Der]
# toral elements: enumerate all 2^dimDer combinations if small
n = len(Dm); tor = []
if n <= 22:
    cur = np.zeros((d, d), np.uint8); prev = 0
    for g in range(1, 1 << n):
        b = (g & -g).bit_length() - 1
        cur ^= Dm[b]
        gray = g ^ (g >> 1)
    # simpler: direct loop over Gray code
    cur = np.zeros((d, d), np.uint8); code = 0
    for g in range(1, 1 << n):
        b = (g & -g).bit_length() - 1
        cur ^= Dm[b]
        if not (mm(cur, cur) ^ cur).any(): tor.append(cur.copy())
    print('toral elements in Der (nonzero):', len(tor))
    # max rank of a torus = log2(1 + max size of commuting family closed under +)
    # greedy: find commuting sets spanning tori
    best = 0
    import random
    random.seed(1)
    for trial in range(2000):
        T = []
        order = random.sample(range(len(tor)), len(tor))
        for i in order:
            t = tor[i]
            if all(not (mm(t, s) ^ mm(s, t)).any() for s in T):
                cand = T + [t]
                if rank(np.array([x.reshape(-1) for x in cand])) == len(cand):
                    T = cand
        best = max(best, len(T))
    print('largest torus found in Der: rank', best)
