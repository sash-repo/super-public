#!/usr/bin/env python3
"""The prolong (g_-, gl(2))_{*,N} for several shearing vectors N."""
import sys, itertools
from mel5 import *
mats, forms = setup()
B = forms[0]
gm2 = [mono(0, 0, 0, 0, 0)]
gm1 = [{lin_index(n): 1} for n in range(4)]
for Nv in [(2,1,1,1,1),(1,2,1,1,2),(1,1,2,2,1),(2,2,1,1,2),(2,1,2,2,1),(1,2,2,2,2)]:
    lim = tuple(5 ** a for a in Nv)
    K = Contact(B, lim)
    H = [quad_hamiltonian(K, A) for A in mats[:3]]
    g0 = [mono(1, 0, 0, 0, 0)] + H
    comps = prolong(K, gm2, gm1, g0, kmax=200)
    dims = {d: len(v) for d, v in comps.items()}
    print(Nv, "total", sum(dims.values()), "top degree", max(dims), flush=True)
