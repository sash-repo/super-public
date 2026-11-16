#!/usr/bin/env python3
"""Dump structure-constant tables (numpy, mod 3) of the algebras compared in iso6.py."""
import numpy as np
from deform import cocycles, deform, g
from er3 import build_gr
P = 3
def tab(alg):
    n = alg.n
    C = np.zeros((n, n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k, c in alg.c[i][j].items():
                C[i, j, k] = c % P
    return C
out = {}
out["k1"] = tab(g)
def co(w, k, lam):
    v = [0]*len(cocycles)
    v[[i for i,(ww,kk,c) in enumerate(cocycles) if ww==w and kk==k][0]] = lam
    return v
out["E6_1"] = tab(deform(co((-6,0),1,1)))
out["E6_2"] = tab(deform(co((-6,0),1,2)))
out["erdef_p"] = tab(deform(co((-3,3),2,1)))
out["erdef_m"] = tab(deform(co((-3,-3),2,1)))
gl, G, dims = build_gr()
out["er_gr"] = tab(G)
np.savez("iso6_tables.npz", **out)
print({k: v.shape for k, v in out.items()})
