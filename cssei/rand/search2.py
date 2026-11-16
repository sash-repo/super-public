#!/usr/bin/env python3
"""Second search: ambients = the simple cores of the BGLLS algebras (tabs/*.npz, with
their Z-grading) and the new 30-dim algebra.  Samplers: sparse sums of basis elements;
'graded': x in g_{-1} (random nonzero), y a sparse element of positive degree;
'homog': x in g_i, y in g_j for random degrees.  Output runs2.jsonl, found/ as search.py.
usage: search2.py SECONDS [workers] [seed0]"""
import sys, os, time, json, glob
import numpy as np
from multiprocessing import Pool
import search as S
from lie import TabAlg, from_css_tab

HERE = S.HERE

def make():
    ams = []
    for p in sorted(glob.glob(os.path.join(HERE, 'tabs', '*.npz'))):
        z = np.load(p)
        a = TabAlg(z['ad'], os.path.basename(p)[:-4]); a.deg = z['deg']
        a.basis = np.eye(a.d, dtype=np.uint8)
        ams.append(a)
    d = json.load(open(os.path.join(HERE, 'new30.json')))   # the 30-dim simple core of the prolong of (V_8, gs(2))
    a = TabAlg(from_css_tab(d['tab']), 'new30'); a.deg = None; a.basis = np.eye(a.d, dtype=np.uint8)
    ams.append(a)
    return ams

def comb(a, idx, rng, k):
    k = min(k, len(idx))
    sel = rng.choice(idx, size=k, replace=False)
    return (a.basis[sel].sum(axis=0) & 1).astype(np.uint8)

def gens_for(a, rng):
    mode = 'sparse'
    if a.deg is not None:
        mode = str(rng.choice(['sparse', 'graded', 'homog', 'trans']))
    k = int(rng.choice([1, 2, 3, 4, 6]))
    if mode == 'sparse':
        g = 2 if rng.random() < 0.8 else 3
        return mode, [comb(a, np.arange(a.d), rng, int(rng.integers(1, k + 1))) for _ in range(g)]
    degs = sorted(set(a.deg.tolist()))
    if mode == 'graded':
        neg = np.nonzero(a.deg == -1)[0]; pos = np.nonzero(a.deg >= 1)[0]
        x = comb(a, neg, rng, int(rng.integers(1, len(neg) + 1)))
        return mode, [x, comb(a, pos, rng, k)]
    if mode == 'trans':      # all of the lowest component + one or two sparse elements of degree >= 0
        low = np.nonzero(a.deg == min(degs))[0]
        nn = np.nonzero(a.deg >= 0)[0]
        return mode, [a.basis[t] for t in low] + [comb(a, nn, rng, k) for _ in range(int(rng.integers(1, 3)))]
    i, j = rng.choice(degs, size=2)
    return mode, [comb(a, np.nonzero(a.deg == i)[0], rng, k), comb(a, np.nonzero(a.deg == j)[0], rng, k)]

def init():
    S.AMS = make()

def trial(seed):
    rng = np.random.default_rng(seed)
    a = S.AMS[seed % len(S.AMS)]
    mode, gens = gens_for(a, rng)
    return S.run_gens(a, mode, gens, rng, seed)

if __name__ == '__main__':
    budget = float(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    log = open(os.path.join(HERE, 'runs2.jsonl'), 'a')
    t0 = time.time(); n = 0
    with Pool(workers, initializer=init) as P:
        for r in P.imap_unordered(trial, range(start, start + 10 ** 7), chunksize=1):
            log.write(json.dumps(r) + '\n'); log.flush(); n += 1
            if time.time() - t0 > budget:
                P.terminate(); break
    print('trials', n)
