#!/usr/bin/env python3
"""For the scan entries (chi(N+)=0 != chi(N-)) with a quotient of dim <= 64: is the quotient
irreducible (1-dim N+-singular space)?  Records the rigorously irreducible dims."""
import numpy as np, json, time
from multiprocessing import Pool
from gfla import GF
from lalg import *
from induce import Induced
from quot import quotient
from hw import H
from scan import chambers
F = GF(4)
CH = {len([k for k in range(N) if k not in H and k not in pos]): None for pos in chambers()}
# the scan used the chamber representatives up to u1<->u2 in this order:
def reps():
    ch = chambers()
    sw = {k: [j for j in range(N) if tuple(WTS[j]) == (WTS[k][1], WTS[k][0])][0] for k in range(N)}
    out = []
    for pos in ch:
        img = tuple(sorted(sw[k] for k in pos))
        if img not in [tuple(sorted(r)) for r in out]: out.append(pos)
    return out
REPS = reps()
def job(args):
    ci, patt, lam = args
    pos = list(REPS[ci]); neg = [k for k in range(N) if k not in H and k not in pos]
    chi = np.zeros(N, dtype=np.int64)
    for i, h in enumerate(H): chi[h] = F.sqrt(F.T[lam[i], lam[i]] ^ lam[i])
    for b, k in enumerate(neg): chi[k] = (patt >> b) & 1
    Q = quotient(F, Induced(F, H + pos, list(lam) + [0] * len(pos), chi).mats())
    S = F.nullspace(np.vstack([Q[k] for k in pos]))
    return (ci, patt, lam, Q[0].shape[0], len(S))
if __name__ == '__main__':
    res = json.load(open('scan.json'))
    # re-derive: rerun every chamber rep / pattern / lambda whose scan quotient was <= 64
    tasks = []
    for ci, pos in enumerate(REPS):
        m = N - 2 - len(pos)
        for patt in range(1, 1 << m):
            for lam in [(0, 0), (0, 1), (1, 0), (1, 1), (5, 11)]:
                tasks.append((ci, patt, lam))
    want = {(r[0], r[1], tuple(r[2])) for r in res if r[3] <= 64}
    tasks = [t for t in tasks if (len([k for k in range(N) if k not in H and k not in REPS[t[0]]]), t[1], t[2]) in want]
    print('tasks', len(tasks), flush=True)
    t0 = time.time(); out = []
    with Pool(10) as P:
        for r in P.imap_unordered(job, tasks, chunksize=2):
            out.append(r)
    json.dump(out, open('scanverify.json', 'w'))
    from collections import Counter
    print('irreducible (singular 1):', sorted(Counter(r[3] for r in out if r[4] == 1).items()))
    print('not proven irreducible:', sorted(Counter(r[3] for r in out if r[4] != 1).items()), round(time.time() - t0), 's')
