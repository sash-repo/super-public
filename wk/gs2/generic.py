#!/usr/bin/env python3
"""Subsec. 7.1, 1) and Statement 7.1.1, 1): baby Verma modules Z_chi(lambda) with lambda generic (chi(h) != 0), both kinds of Borel;
irreducibility by the singular vectors (N+ acts nilpotently since chi(N+)=0)."""
import numpy as np, itertools, sys
from gfla import GF
from lalg import *
from induce import check_rep
from hw import module, singular, radical_quotient_dim, borel
F = GF(4); rng = np.random.default_rng(3)
for phi in [(1, 10), (10, 1), (-1, -10), (-10, -1), (1, -10), (-1, 10)]:
    pos, neg = borel(phi)
    res = []
    for t in range(3):
        lam = tuple(int(a) for a in rng.integers(2, F.q, 2))
        Mod, chi, pos, neg = module(F, phi, lam)
        M = Mod.mats()
        if t == 0: check_rep(F, M, chi)
        S = singular(F, M, pos)
        q = radical_quotient_dim(F, M, range(N)) if len(S) > 1 else Mod.dim
        res.append((lam, Mod.dim, len(S), q))
    print('phi', phi, 'dim N- =', len(neg), [NAMES[k] for k in neg if '^' in NAMES[k]], res, flush=True)
