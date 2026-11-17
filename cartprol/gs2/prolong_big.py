#!/usr/bin/env python3
"""g_1 of the Cartan prolong for a large g_{-1}: the kernel is updated block by block (pairs (i,j) with
i fixed), so the matrix never exceeds (rows of one block) x (current kernel dim).
In the text: Subsec. 4.5 (the 64-dimensional restricted module has trivial prolong)."""
import numpy as np, pickle, sys
from f2lin import nullspace, rref
from lalg import *
def g1(g0mats, n, odd=()):
    A = np.array([np.asarray(m, dtype=np.uint8).flatten() & 1 for m in g0mats], dtype=np.uint8)
    R, piv = rref(A); ev = [r.reshape(n, n) for r in R]; m = len(ev)
    K = np.eye(n * m, dtype=np.uint8)          # current kernel basis (rows)
    for i in range(n):
        rows = []
        for j in range(i + 1, n):
            blk = np.zeros((n, n * m), dtype=np.uint8)
            for b in range(m):
                blk[:, i * m + b] ^= ev[b][:, j]; blk[:, j * m + b] ^= ev[b][:, i]
            rows.append(blk)
        if i in odd:
            blk = np.zeros((n, n * m), dtype=np.uint8)
            for b in range(m): blk[:, i * m + b] ^= ev[b][:, i]
            rows.append(blk)
        if not rows: continue
        Msys = np.vstack(rows)
        P = (Msys.astype(np.int64) @ K.T.astype(np.int64)) & 1     # rows x dimK
        C = nullspace(P.astype(np.uint8), len(K))
        K = ((C.astype(np.int64) @ K.astype(np.int64)) & 1).astype(np.uint8)
        if len(K) == 0: break
    return m, len(K)
if __name__ == '__main__':
    mods = pickle.load(open('restricted_irreps.pkl', 'rb'))
    M = mods[(1, 1)]; n = M[0].shape[0]; Id = np.eye(n, dtype=np.int64)
    for nm, g0 in [('gs(2)', [M[k] for k in range(NG)]), ('gs(2)+K1', [M[k] for k in range(NG)] + [Id]),
                   ('L', list(M)), ('L+K1', list(M) + [Id])]:
        print('V64, g0 = %-9s dim g0 = %d, dim g_1: V %d, Pi(V) %d' % ((nm,) + g1(g0, n)[:1] + (g1(g0, n)[1], g1(g0, n, range(n))[1])), flush=True)
