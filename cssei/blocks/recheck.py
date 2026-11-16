#!/usr/bin/env python3
"""Recheck of the random search: the random search kept ONE table per fingerprint, so constituents with
equal fingerprints were never compared.  Re-run every seed that produced a simple constituent of
dim 16..31 and compute the CSS invariants of EACH such constituent.  Output recheck.jsonl."""
import os, sys, json
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rand')); sys.path.insert(0, os.path.join(HERE, '..', 'cmp'))
import search as S, search2 as S2
from lie import closure, table_of, factors
from count import invariants

def tabify(ad):
    n = ad.shape[0]
    return [[int(sum(int(ad[i][k, j]) << k for k in range(n))) for j in range(n)] for i in range(n)]

AMS1 = AMS2 = None
def init():
    global AMS1, AMS2
    AMS1 = S.make_ambients(); AMS2 = S2.make()

def work(job):
    fn, seed = job
    rng = np.random.default_rng(seed)
    if fn == 'runs.jsonl':
        amb = AMS1[seed % len(AMS1)]
        mode = 'graded' if (amb.deg is not None and rng.random() < 0.5) else 'sparse'
        import ambients as A
        if mode == 'graded':
            gens = A.graded_pair(amb, rng, kmax=int(rng.integers(1, 5)))
        else:
            g = 2 if rng.random() < 0.8 else 3
            kmax = int(rng.choice([1, 2, 3, 4, 6]))
            gens = [A.sparse(amb, rng, kmax) for _ in range(g)]
    else:
        amb = AMS2[seed % len(AMS2)]
        mode, gens = S2.gens_for(amb, rng)
    E = closure(amb, gens, amb.L)
    ad = table_of(amb, E)
    out = []; factors(ad, rng, out)
    res = []
    for fad, st, AK in out:
        d = fad.shape[0]
        if st == 'simple' and 16 <= d <= 31:
            inv = invariants(tabify(fad))
            res.append({'d': d, 'inv': [int(inv[0]), int(inv[1])]})
    return {'file': fn, 'seed': seed, 'amb': amb.name, 'dimU': E.dim(), 'res': res}

if __name__ == '__main__':
    jobs = []
    for fn in ('runs.jsonl', 'runs2.jsonl'):
        for line in open(os.path.join(HERE, '..', 'rand', fn)):
            r = json.loads(line)
            if any(f['status'] == 'simple' and 16 <= f['d'] <= 31 for f in r['factors']):
                jobs.append((fn, r['seed']))
    done = set()
    outp = os.path.join(HERE, 'recheck.jsonl')
    if os.path.exists(outp):
        for line in open(outp):
            r = json.loads(line); done.add((r['file'], r['seed']))
    jobs = [j for j in jobs if j not in done]
    print('jobs', len(jobs), flush=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with Pool(nw, initializer=init) as P, open(outp, 'a') as f:
        for i, r in enumerate(P.imap_unordered(work, jobs, chunksize=1)):
            f.write(json.dumps(r) + '\n'); f.flush()
            if i % 200 == 0: print(i, flush=True)
