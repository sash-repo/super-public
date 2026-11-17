"""Subsec. 7.1, 2) of the text: the stabilizer of a generic chi in L^* (dim 3) is p-closed and its
semisimple part has rank 2, so it is a 2-dim torus plus a nilpotent line; not toral.  Run after shenL.py."""
import numpy as np
from gfla import GF
from lalg import *
F = GF(8); rng = np.random.default_rng(7)
for t in range(6):
    chi = rng.integers(0, F.q, N)
    S = F.nullspace(Bchi(F, chi))
    # p-map restricted to S: is S closed? compute semisimple span
    closed = all(F.rank(np.vstack([S, psq(F, s)[None]])) == len(S) for s in S)
    y = F.dot(rng.integers(0, F.q, (1, len(S))), S)[0]
    seq = [y]
    for k in range(60): seq.append(psq(F, seq[-1]))
    r = F.rank(np.array(seq[30:60]))
    # nilpotent elements: p-map kills after some steps? dim of {s: s^[2^k]=0}
    nil = []
    for s in S:
        z = s
        for k in range(20): z = psq(F, z)
        nil.append(not z.any())
    print('stab dim', len(S), 'p-closed', closed, 'semisimple rank', r, 'nilpotent basis vecs', sum(nil))
