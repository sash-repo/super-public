#!/usr/bin/env python3
"""In the text: Table (3) (prolongs of V_7 and of the adjoint module)."""
import numpy as np, pickle, sys
from f2lin import rank
from lalg import *
from cprolong import prolong
from gfla import GF
mods = pickle.load(open('restricted_irreps.pkl', 'rb'))
F1 = GF(1)
Din = np.eye(N, dtype=np.int64)[NAMES.index('u1d1')]; Dout = np.load('outerD.npy')
E = np.eye(N, dtype=np.int64)
def comb(M, x):
    out = np.zeros_like(M[0])
    for j in np.nonzero(x)[0]: out ^= M[j]
    return out & 1
def parity(M, D):
    """odd basis vectors: eigenvalue-1 eigenvectors of rho(D); returns change of basis to an
    eigenbasis and the list of odd indices"""
    R = comb(M, D); n = R.shape[0]
    ev = F1.nullspace(R); od = F1.nullspace(R ^ np.eye(n, dtype=np.int64))
    P = np.vstack([ev, od]).T          # columns = new basis
    assert rank(P) == n
    return P, list(range(len(ev), n))
def conj(M, P):
    # matrices in the new basis: P^{-1} X P
    n = P.shape[0]
    Pinv = F1.nullspace(np.hstack([P, np.eye(n, dtype=np.int64)]))  # not used
    from f2lin import rref
    aug = np.hstack([P.astype(np.uint8), np.eye(n, dtype=np.uint8)]); R, piv = rref(aug)
    Pi = R[:, n:].astype(np.int64)
    return [(Pi @ X @ P) & 1 for X in M]
if __name__ == '__main__':
    maxk = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for lam, nm in [((1, 0), 'V7'), ((0, 1), 'ad14')]:
        M = mods[lam]; n = M[0].shape[0]
        Id = np.eye(n, dtype=np.int64)
        gs = [M[k] for k in range(NG)]; Lm = [M[k] for k in range(N)]
        print('%s: dim %d, dim rho(gs2) = %d, dim rho(L) = %d' % (nm, n, rank(np.array([m.flatten() for m in gs])),
              rank(np.array([m.flatten() for m in Lm]))), flush=True)
        for gname, g0 in [('gs(2)', gs), ('gs(2)+K1', gs + [Id]), ('L', Lm), ('L+K1', Lm + [Id])]:
            print('   g0 = %-9s  g_{-1}=V: %s   Pi(V): %s' % (gname, prolong(g0, n, (), maxk),
                  prolong(g0, n, range(n), maxk)), flush=True)
        # superizations: g0 = s (parity from D), g_{-1} = V with its parity, and Pi(V)
        for sname, D, sub in [('s_a', Din, [k for k in range(N) if NAMES[k] != '(u2)^2']), ('s_b', Dout, list(range(N)))]:
            P, odd = parity(M, D)
            Mc = conj(M, P)
            g0 = [Mc[k] for k in sub]
            ev = [i for i in range(n) if i not in odd]
            for g0n, g0x in [(sname, g0), (sname + '+K1', g0 + [Id])]:
                print('   g0 = %-9s  g_{-1}=V (sdim (%d|%d)): %s   Pi(V): %s' % (g0n, len(ev), len(odd),
                      prolong(g0x, n, odd, maxk), prolong(g0x, n, ev, maxk)), flush=True)
