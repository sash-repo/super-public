#!/usr/bin/env python3
"""The 8-dimensional irreducible gs(2)-modules with chi(N+)=0 != chi(N-) (chi on u1 only, etc.),
over F_2; their Cartan prolongs with g_0 = gs(2), c(gs(2)), and invariant forms.
In the text: Subsec. 4.5 (the prolongs of the modules V_8: dims 14 (15), 8, 1)."""
import numpy as np, sys, pickle
from f2lin import rank, nullspace
from gfla import GF
from lalg import *
from induce import Induced, check_rep
from quot import quotient
from hw import H
from scanverify import REPS
from cprolong import prolong
F = GF(1)
mods8 = {}
for (ci, patt, lam) in [(1, 16, (1, 1)), (1, 24, (1, 1)), (3, 16, (0, 1)), (4, 1, (0, 1)), (4, 9, (0, 1))]:
    pos = list(REPS[ci]); neg = [k for k in range(N) if k not in H and k not in pos]
    chi = np.zeros(N, dtype=np.int64)
    for b, k in enumerate(neg): chi[k] = (patt >> b) & 1
    Q = quotient(F, Induced(F, H + pos, list(lam), chi).mats() if False else Induced(F, H + pos, list(lam) + [0] * len(pos), chi).mats())
    check_rep(F, Q, chi)
    S = F.nullspace(np.vstack([Q[k] for k in pos]))
    n = Q[0].shape[0]; Id = np.eye(n, dtype=np.int64)
    gs = [Q[k] for k in range(NG)]
    print('chi on', [NAMES[k] for k in range(N) if chi[k]], 'lam', lam, 'dim', n, 'singular', len(S),
          'rank rho(gs2)', rank(np.array([m.flatten() for m in gs], dtype=np.uint8)),
          ' prolong V:', prolong(gs, n, (), 6), 'Pi(V):', prolong(gs, n, range(n), 6),
          ' c: V', prolong(gs + [Id], n, (), 6), 'Pi(V)', prolong(gs + [Id], n, range(n), 6), flush=True)
    mods8[(ci, patt, lam)] = (Q, chi)
pickle.dump(mods8, open('mods8.pkl', 'wb'))
