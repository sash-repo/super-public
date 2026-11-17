#!/usr/bin/env python3
"""Scan: irreducible quotients of Z_chi(lambda) for chi(N+)=0, chi|N- in F_2^{N-} (all patterns),
lambda in F_2^2 and one generic lambda, over every chamber (Borel) up to u1<->u2.  Records dims."""
import numpy as np, itertools, sys, json, time
from multiprocessing import Pool
from gfla import GF
from lalg import *
from induce import Induced
from hw import singular, H
from quot import quotient
F = GF(4)
def chambers():
    seen = {}
    for t in range(720):
        th = 2 * np.pi * (t + 0.5) / 720
        phi = (np.cos(th), np.sin(th))
        pos = tuple(k for k in range(N) if k not in H and phi[0] * WTS[k][0] + phi[1] * WTS[k][1] > 0)
        seen.setdefault(pos, phi)
    return seen
def job(args):
    pos, patt, lam = args
    pos = list(pos); neg = [k for k in range(N) if k not in H and k not in pos]
    chi = np.zeros(N, dtype=np.int64)
    for i, h in enumerate(H): chi[h] = F.sqrt(F.T[lam[i], lam[i]] ^ lam[i])
    for b, k in enumerate(neg): chi[k] = (patt >> b) & 1
    Mod = Induced(F, H + pos, list(lam) + [0] * len(pos), chi)
    M = Mod.mats()
    S = singular(F, M, pos)
    q = Mod.dim if len(S) == 1 else quotient(F, M)[0].shape[0]
    return (len(neg), patt, lam, q)
if __name__ == '__main__':
    ch = chambers()
    print('chambers:', len(ch), [len(p) for p in ch])
    # up to u1<->u2 (swap of the two coordinates of the weights)
    sw = {k: [j for j in range(N) if tuple(WTS[j]) == (WTS[k][1], WTS[k][0])][0] for k in range(N)}
    reps = []
    for pos in ch:
        img = tuple(sorted(sw[k] for k in pos))
        if img not in [tuple(sorted(r)) for r in reps]: reps.append(pos)
    print('chambers up to u1<->u2:', len(reps))
    tasks = []
    for pos in reps:
        m = N - 2 - len(pos)
        for patt in range(1, 1 << m):
            for lam in [(0, 0), (0, 1), (1, 0), (1, 1), (5, 11)]:
                tasks.append((pos, patt, lam))
    print('modules:', len(tasks), flush=True)
    t0 = time.time(); res = []
    with Pool(10) as P:
        for r in P.imap_unordered(job, tasks, chunksize=4):
            res.append(r)
            if len(res) % 200 == 0: print(len(res), round(time.time() - t0), flush=True)
    json.dump([list(map(lambda x: x if not isinstance(x, tuple) else list(x), r)) for r in res], open('scan.json', 'w'))
    from collections import Counter
    print('dims of the irreducible quotients:', sorted(Counter(r[3] for r in res).items()))
