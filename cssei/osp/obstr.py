"""Second-order obstructions: Ob(lam) = sum_{i,j} lam_i lam_j R(C_i, C_j, Q_j) modulo im(delta),
C_i the representatives of H^2_ev.  Output: the quadrics (over F_2) cutting out the unobstructed cone."""
import numpy as np, itertools, pickle, sys
from sconst import load, NAME
from deform import Deform
from f2lin import rref

def reducer(D):
    Rm, piv = rref(D.T)             # rows span im(delta)
    return Rm, piv
def reduce(v, Rm, piv):
    v = v.copy()
    for r, p in zip(Rm, piv):
        if v[p]: v ^= r
    return v

def analyse(t):
    Df = Deform(*load(t)[:3]); Df.cohomology()
    Rm, piv = reducer(Df.D)
    H = Df.H; h = len(H)
    CQ = [Df.unpack(v) for v in H]
    W = {}
    for i in range(h):
        for j in range(i, h):
            if i == j: w = Df.R(CQ[i][0], CQ[i][0], CQ[i][1])
            else: w = (Df.R(CQ[i][0], CQ[j][0], CQ[j][1]) + Df.R(CQ[j][0], CQ[i][0], CQ[i][1])) & 1
            W[(i, j)] = reduce(w.astype(np.uint8), Rm, piv)
    # coordinates of coker: columns where some W is nonzero
    keys = sorted(W)
    M = np.array([W[k] for k in keys], dtype=np.uint8)      # monomials x eq-coords
    cols = np.nonzero(M.any(axis=0))[0]
    M = M[:, cols]
    # quadrics = columns of M, i.e. for each coker coordinate c: sum_k M[k,c] mono_k
    Q = rref(M.T)[0]                                         # independent quadrics
    print(NAME[t], 'h =', h, ' independent obstruction quadrics:', len(Q))
    return Df, keys, Q

def f2_points(h, keys, Q):
    sols = []
    for lam in itertools.product([0, 1], repeat=h):
        if not any(lam): continue
        mono = np.array([lam[i] & lam[j] for (i, j) in keys], dtype=np.uint8)
        if not ((Q @ mono) & 1).any(): sols.append(lam)
    return sols

if __name__ == '__main__':
    out = {}
    for t in ('inner', 'outer'):
        Df, keys, Q = analyse(t)
        sols = f2_points(len(Df.H), keys, Q)
        print('   unobstructed (to order 2) nonzero lam over F_2:', len(sols))
        out[t] = dict(H=Df.H, keys=keys, Q=Q, sols=sols)
    pickle.dump(out, open('obstr.pkl', 'wb'))
