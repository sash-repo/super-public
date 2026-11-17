#!/usr/bin/env python3
"""Derived series and simplicity of the finite prolongs; export of the simple core.
In the text: Theorem 4.6, 1)-2) (the 28-dimensional core, absolutely simple)."""
import numpy as np, pickle, sys, json, glob
from f2lin import rref, rank, nullspace, solve
from buildalg import assoc_dim
def sub_table(C, S):
    """structure constants of the subalgebra spanned by rows of S (in its own coordinates)"""
    k = len(S); Ci = C.astype(np.int64)
    T = np.zeros((k, k, k), dtype=np.uint8)
    for a in range(k):
        for b in range(k):
            v = np.einsum('i,j,ijk->k', S[a].astype(np.int64), S[b].astype(np.int64), Ci) & 1
            c = solve(S.T, v); assert c is not None, 'not closed'
            T[a, b] = c
    return T
def derived(T):
    k = T.shape[0]
    V = T.reshape(k * k, k)
    return rref(V)[0]
def series(C):
    out = []; T = C
    while True:
        D = derived(T)
        out.append(len(D))
        if len(D) == T.shape[0] or len(D) == 0: return out, T
        T = sub_table(T, D)
if __name__ == '__main__':
    for f in sorted(glob.glob('alg_*.pkl')):
        C, dims = pickle.load(open(f, 'rb'))
        ser, core = series(C)
        k = core.shape[0]
        a = assoc_dim(core)
        print('%-26s dims %s  derived series %s  core dim %d  assoc(ad core) %d = %d^2? %s' % (f, dims, ser, k, a, k, a == k * k), flush=True)
        tab = [[int(''.join(map(str, core[i, j][::-1])), 2) for j in range(k)] for i in range(k)]
        json.dump({'name': f, 'dim': k, 'tab': tab}, open(f.replace('.pkl', '_core.json'), 'w'))
