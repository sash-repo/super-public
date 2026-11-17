"""Statement 7.1.1, 3): sdim L = (9|8) for the gradings of both superizations s_a (inner, by u1d1)
and s_b (outer); the stabilizers of a generic chi vanishing on L_od.  Run after shenL.py, toral.py."""
import numpy as np
from gfla import GF
from lalg import *
F8 = GF(8); F1 = GF(1); rng = np.random.default_rng(11)
Din = np.eye(N, dtype=np.int64)[NAMES.index('u1d1')]; Dout = np.load('outerD.npy')
for nm, D in [('inner u1d1', Din), ('outer', Dout)]:
    A = admat(F1, D)
    od = F1.nullspace(A ^ np.eye(N, dtype=np.int64))      # L_od basis (rows)
    ev = F1.nullspace(A)
    rs = []
    for t in range(4):
        # chi vanishing on L_od: chi = random combination of the annihilator of L_od
        ann = F1.nullspace(od)          # functionals (rows) killing L_od
        chi = F8.dot(rng.integers(0, F8.q, (1, len(ann))), ann)[0]
        rs.append(N - F8.rank(Bchi(F8, chi)))
    print(nm, 'sdim L = (%d|%d)' % (len(ev), len(od)), ' generic stabilizer dims for chi|L_od=0:', rs)
