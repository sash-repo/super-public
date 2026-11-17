#!/usr/bin/env python3
"""Statement 7.1.1, 3): the restricted irreducible modules over the superizations, with their
superdimensions with respect to the grading elements of s_a and s_b.  Writes restricted_irreps.pkl
(read by verifyq.py).  Run after shenL.py, toral.py."""
import numpy as np, itertools, sys, pickle
from gfla import GF
from lalg import *
from induce import Induced, check_rep
from hw import borel, singular, H
from quot import quotient
Din = np.eye(N, dtype=np.int64)[NAMES.index('u1d1')]; Dout = np.load('outerD.npy')
def comb(F, M, x):
    out = np.zeros_like(M[0])
    for j in np.nonzero(x)[0]: out ^= F.T[x[j], M[j]]
    return out
def sdim(F, M, D, mu=0):
    R = comb(F, M, D); n = R.shape[0]
    a = n - F.rank(R ^ F.T[mu, np.eye(n, dtype=np.int64)])
    b = n - F.rank(R ^ F.T[mu ^ 1, np.eye(n, dtype=np.int64)])
    return a, b
F1 = GF(1)
pos, neg = borel((1, 10))
mods = {}
for lam in itertools.product(range(2), repeat=2):
    Mod = Induced(F1, H + pos, list(lam) + [0] * len(pos), np.zeros(N, dtype=np.int64))
    M = quotient(F1, Mod.mats()); check_rep(F1, M, np.zeros(N, dtype=np.int64))
    mods[lam] = M
    print('restricted L%s dim %d  sdim wrt inner D: %s  wrt outer D: %s' % (lam, M[0].shape[0], sdim(F1, M, Din), sdim(F1, M, Dout)))
pickle.dump(mods, open('restricted_irreps.pkl', 'wb'))
