#!/usr/bin/env python3
"""Irreducible highest-weight modules L(lambda) over gs(2) (i.e. over u(L,chi), chi(N+-)=0) wrt the
BGLLS Borel (X1+ = u1d2, X2+ = u2vol, phi=(3,2)), lambda = (lambda(H1), lambda(H2)) in GF(16)^2,
H1 = u1d1+u2d2, H2 = u2d2.  Records dim, sdim wrt u1d1 (the grading of s_a), and the character as
multiplicities of the four weight cosets lambda + (i,j), (i,j) in F_2^2.
In the text: Table (2) and Proposition 4.3, 1)."""
import numpy as np, itertools, sys, json, time
from multiprocessing import Pool
from gfla import GF
from lalg import *
from induce import Induced
from hw import singular
from quot import quotient
F = GF(4)
H = [NAMES.index('u1d1'), NAMES.index('u2d2')]
phi = (3, 2)
pos = [k for k in range(N) if k not in H and phi[0] * WTS[k][0] + phi[1] * WTS[k][1] > 0]
def eigdim(A, mu):
    n = A.shape[0]; B = A ^ F.T[mu, np.eye(n, dtype=np.int64)]
    # generalized eigenspace: kernel of B^n (use B^8 enough? use repeated squaring up to n)
    P = B.copy()
    for _ in range(8): P = F.dot(P, P)
    return n - F.rank(P)
def job(l):
    l1, l2 = l          # lambda(H1), lambda(H2)
    lam = [l1 ^ l2, l2]  # lambda(u1d1) = l1 + l2, lambda(u2d2) = l2
    chi = np.zeros(N, dtype=np.int64)
    for i, h in enumerate(H): chi[h] = F.sqrt(F.T[lam[i], lam[i]] ^ lam[i])
    Mod = Induced(F, H + pos, lam + [0] * len(pos), chi)
    M = Mod.mats()
    Q = quotient(F, M); n = Q[0].shape[0]
    RH1 = Q[H[0]] ^ Q[H[1]]; RH2 = Q[H[1]]; RD = Q[H[0]]
    # joint coset multiplicities
    mult = {}
    for i in (0, 1):
        for j in (0, 1):
            A = RH1 ^ F.T[l1 ^ i, np.eye(n, dtype=np.int64)]
            Bm = RH2 ^ F.T[l2 ^ j, np.eye(n, dtype=np.int64)]
            for _ in range(8): A = F.dot(A, A); Bm = F.dot(Bm, Bm)
            mult['%d%d' % (i, j)] = n - F.rank(np.vstack([A, Bm]))
    sd = (eigdim(RD, lam[0]), eigdim(RD, lam[0] ^ 1))
    return (l1, l2, n, sd, mult)
if __name__ == '__main__':
    tasks = list(itertools.product(range(F.q), repeat=2))
    t0 = time.time(); res = []
    with Pool(10) as P:
        for r in P.imap_unordered(job, tasks):
            res.append(r)
    json.dump(res, open('hwscan.json', 'w'))
    print('done in', round(time.time() - t0), 's')
    from collections import Counter
    print(Counter((r[2], tuple(r[3]), tuple(sorted(r[4].items()))) for r in res))
