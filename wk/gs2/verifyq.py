#!/usr/bin/env python3
"""Statement 7.1.1, 1): rigorous irreducibility of the quotients Q used in the tables: N+ acts nilpotently (chi(N+)=0),
so Q is irreducible if and only if its N+-singular vectors form a line (the image of v generates Q)."""
import numpy as np, itertools, pickle
from gfla import GF
from lalg import *
from induce import Induced
from hw import singular
from quot import quotient
from hwscan import pos as posB, H
F1 = GF(1); F = GF(4)
from hw import borel
posA, negA = borel((1, 10))
mods = pickle.load(open('restricted_irreps.pkl', 'rb'))
for lam, Q in mods.items():
    S = F1.nullspace(np.vstack([Q[k] for k in posA]))
    print('restricted', lam, 'dim', Q[0].shape[0], 'singular in quotient:', len(S))
for (l1, l2) in [(0, 0), (0, 1), (1, 0), (1, 1), (0, 5), (5, 0), (5, 4), (5, 7), (7, 9), (3, 12)]:
    lam = [l1 ^ l2, l2]
    chi = np.zeros(N, dtype=np.int64)
    for i, h in enumerate(H): chi[h] = F.sqrt(F.T[lam[i], lam[i]] ^ lam[i])
    Q = quotient(F, Induced(F, H + posB, lam + [0] * len(posB), chi).mats())
    S = F.nullspace(np.vstack([Q[k] for k in posB]))
    print('BGLLS Borel (%d,%d): dim %d, singular in quotient %d' % (l1, l2, Q[0].shape[0], len(S)))
