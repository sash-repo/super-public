#!/usr/bin/env python3
"""Cartan prolong (g_{-1}, g_0)_* over F_2 for g_{-1}=V (dim n), g_0 a subspace of gl(V).
odd = set of indices i of V whose basis vector is odd (then also D(e_i)e_i = 0 is imposed:
Jacobi with the squaring, s(e_i)=0).  Returns [dim g_0, dim g_1, ...]."""
import numpy as np, sys
from f2lin import nullspace, rank

def prolong(g0mats, n, odd=(), maxk=12, verbose=False):
    g0 = [np.asarray(m, dtype=np.uint8) & 1 for m in g0mats]
    # independent basis of g_0
    A = np.array([m.flatten() for m in g0], dtype=np.uint8)
    from f2lin import rref
    R, piv = rref(A); g0 = [r.reshape(n, n) for r in R]
    dims = [len(g0)]
    # ev_prev[b] is an (n x p) matrix: column j = coords of b(e_j) in g_{k-2} (p = its dim), for k-1=0 p=n
    ev = [m.astype(np.uint8) for m in g0]
    p = n
    for k in range(1, maxk + 1):
        m = len(ev)
        if m == 0: break
        # unknown c[i,b], index i*m+b
        rows = []
        for i in range(n):
            for j in range(i + 1, n):
                blk = np.zeros((p, n * m), dtype=np.uint8)
                for b in range(m):
                    blk[:, i * m + b] ^= ev[b][:, j]
                    blk[:, j * m + b] ^= ev[b][:, i]
                rows.append(blk)
            if i in odd:
                blk = np.zeros((p, n * m), dtype=np.uint8)
                for b in range(m): blk[:, i * m + b] ^= ev[b][:, i]
                rows.append(blk)
        Msys = np.vstack(rows)
        Ns = nullspace(Msys, n * m)
        dims.append(len(Ns))
        if verbose: print('  g_%d: %d' % (k, len(Ns)), flush=True)
        # new ev: D(e_j) coords in g_{k-1} = c[j,:]
        ev = [c.reshape(n, m).T.copy() for c in Ns]      # (m x n): column j = c[j,:]
        p = m
    return dims
