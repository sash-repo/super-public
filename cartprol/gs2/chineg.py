#!/usr/bin/env python3
"""Z_chi(lambda) with chi(N+)=0 but chi(N-) random (nonzero): irreducible quotient dims.
In the text: Proposition 4.3, 2) (the 128-dimensional induced module is irreducible)."""
import numpy as np, sys
from gfla import GF
from lalg import *
from induce import Induced, check_rep
from hw import borel, singular, radical_quotient_dim, H
F = GF(4); rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
for phi in [(1, 10), (-1, -10)]:
    pos, neg = borel(phi)
    for t in range(3):
        lam = [int(a) for a in rng.integers(0, F.q, 2)]
        chi = np.zeros(N, dtype=np.int64)
        for i, h in enumerate(H): chi[h] = F.sqrt(F.T[lam[i], lam[i]] ^ lam[i])
        for k in neg: chi[k] = int(rng.integers(0, F.q))
        Mod = Induced(F, H + pos, lam + [0] * len(pos), chi)
        M = Mod.mats()
        if t == 0: check_rep(F, M, chi)
        S = singular(F, M, pos)
        q = radical_quotient_dim(F, M, range(N)) if len(S) > 1 else Mod.dim
        print('phi', phi, 'lam', lam, 'dim Z', Mod.dim, 'sing', len(S), 'irr quotient', q, flush=True)
