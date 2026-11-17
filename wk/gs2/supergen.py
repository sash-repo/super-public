#!/usr/bin/env python3
"""Statement 7.1.1, 3): irreducible u(L,chi)-modules with chi|L_od = 0 (= modules over the superizations) of dim 128:
Z_chi(lambda), Borel phi=(1,10), chi(N+)=0, chi on N- random subject to chi(L_od)=0."""
import numpy as np, sys
from gfla import GF
from lalg import *
from induce import Induced, check_rep
from hw import borel, singular, H
from superrep import sdim, Din, Dout
F1 = GF(1); F = GF(8); rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
pos, neg = borel((1, 10))
E = np.eye(N, dtype=np.int64)
for nm, D in [('s_a (inner D=u1d1)', Din), ('s_b (outer D)', Dout)]:
    A = admat(F1, D)
    od = F1.nullspace(A ^ E)
    Wsp = np.vstack([od] + [E[k] for k in pos])
    ann = F1.nullspace(Wsp)                       # chi in ann: chi(L_od)=chi(N+)=0
    print(nm, ': dim of the admissible chi-space', len(ann), ' chi(h) free?',
          F1.rank(ann[:, H]))
    done = 0
    for t in range(40):
        chi = F.dot(rng.integers(0, F.q, (1, len(ann))), ann)[0]
        lam = []
        for h in H:
            c = F.T[chi[h], chi[h]]
            r = [x for x in range(F.q) if F.T[x, x] ^ x == c]
            if not r: break
            lam.append(r[0])
        if len(lam) < 2: continue
        Mod = Induced(F, H + pos, lam + [0] * len(pos), chi)
        M = Mod.mats()
        if done == 0: check_rep(F, M, chi)
        S = singular(F, M, pos)
        mu = [x for x in range(F.q) if F.T[x, x] ^ x == F.T[chi[D != 0].max() if False else 0, 0]]
        # eigenvalues of rho(D): mu, mu+1 with mu^2+mu = chi(D)^2
        cD = 0
        for k in np.nonzero(D)[0]: cD ^= chi[k]
        cc = F.T[cD, cD]
        mu = [x for x in range(F.q) if F.T[x, x] ^ x == cc][0]
        print('   dim Z', Mod.dim, ' singular', len(S), ' sdim wrt D:', sdim(F, M, D, mu), flush=True)
        done += 1
        if done == 2: break
