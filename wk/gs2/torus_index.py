"""Subsec. 7.1, 2) and the row gs(2) of Table (16): the largest tori of L are of dim 3, ind L = 3,
and the generic stabilizer is abelian with semisimple rank 2 < 3 (not a torus).  Run after shenL.py."""
import numpy as np, sys
from gfla import GF
from lalg import *
F = GF(8); rng = np.random.default_rng(1)
# maximal torus dimension: span of x^[2^k], k>=K, for generic x
for trial in range(3):
    x = rng.integers(0, F.q, N)
    seq = [x]
    for k in range(40): seq.append(psq(F, seq[-1]))
    print('torus dim from generic x:', F.rank(np.array(seq[20:40])))
# index: min dim of stabilizer of chi (generic chi)
for trial in range(3):
    chi = rng.integers(0, F.q, N)
    print('generic chi: rank B_chi =', F.rank(Bchi(F, chi)), ' stabilizer dim', N - F.rank(Bchi(F, chi)))
# stabilizer of generic chi: is it a torus?
chi = rng.integers(0, F.q, N)
S = F.nullspace(Bchi(F, chi))
print('stab dim', len(S))
# check abelian and toral-closure: span of p-powers of a generic element of S
y = F.dot(rng.integers(0, F.q, (1, len(S))), S)[0]
seq=[y]
for k in range(40): seq.append(psq(F, seq[-1]))
print('rank of {y^[2^k], k>=20}:', F.rank(np.array(seq[20:40])), ' abelian:',
      all(not bracket(F, S[i], S[j]).any() for i in range(len(S)) for j in range(len(S))))
