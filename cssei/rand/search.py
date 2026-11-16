#!/usr/bin/env python3
"""Eick-style random search: random pairs (or triples) in an ambient, the subalgebra
U they generate, the simple composition factors of U.  Each factor is recorded with
cheap invariants; its table is kept (found/*.npz) the first time its fingerprint occurs.
usage: search.py SECONDS [workers]"""
import sys, os, time, json, hashlib
import numpy as np
from multiprocessing import Pool
from lie import closure, table_of, factors, centroid_dim, restricted, TabAlg, from_css_tab
from gf2 import mm, rank
import ambients as A

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'found'), exist_ok=True)

def profile(ad, rng, n=64):
    d = ad.shape[0]
    rs = []
    for _ in range(n):
        x = rng.integers(0, 2, d).astype(np.uint8)
        adx = (ad[np.nonzero(x)[0]].sum(axis=0) & 1).astype(np.uint8)
        rs.append(rank(adx))
    return max(rs), min(rs)

def make_ambients():
    ams = []
    for n in (8, 9, 10, 11, 12, 13):
        ams.append(A.sl(n))
    for m in (4, 5, 6):
        ams.append(A.witt(m))
    for Ns in ((2, 2), (3, 2), (2, 1, 1), (2, 2, 1), (3, 1, 1), (2, 1, 1, 1), (2, 2, 2), (3, 2, 1), (2, 2, 1, 1), (3, 3)):
        ams.append(A.wittN(Ns))
    return ams

AMS = None
def init():
    global AMS
    AMS = make_ambients()

def trial(seed):
    rng = np.random.default_rng(seed)
    amb = AMS[seed % len(AMS)]
    mode = 'graded' if (amb.deg is not None and rng.random() < 0.5) else 'sparse'
    if mode == 'graded':
        gens = A.graded_pair(amb, rng, kmax=int(rng.integers(1, 5)))
    else:
        g = 2 if rng.random() < 0.8 else 3
        kmax = int(rng.choice([1, 2, 3, 4, 6]))
        gens = [A.sparse(amb, rng, kmax) for _ in range(g)]
    return run_gens(amb, mode, gens, rng, seed)

def run_gens(amb, mode, gens, rng, seed):
    t0 = time.time()
    E = closure(amb, gens, amb.L)
    ad = table_of(amb, E)
    out = []
    nab = factors(ad, rng, out)
    res = []
    for (fad, st, AK) in out:
        d = fad.shape[0]
        c = centroid_dim(fad, AK) if st == 'simple' else None
        r = restricted(fad) if d <= 160 else None
        mx, mn = profile(fad, rng)
        fp = '%d:%s:%s:%s:%d' % (d, c, r, st, mx)
        res.append({'d': d, 'status': st, 'centroid': c, 'restricted': r, 'maxrank': mx, 'minrank': mn, 'fp': fp})
        path = os.path.join(HERE, 'found', hashlib.md5(fp.encode()).hexdigest()[:12] + '.npz')
        if not os.path.exists(path):
            np.savez_compressed(path, ad=fad, fp=fp)
    return {'seed': int(seed), 'amb': amb.name, 'mode': str(mode), 'ngens': len(gens), 'dimU': E.dim(),
            'abelian': int(nab), 'factors': res, 'sec': round(time.time() - t0, 2)}

if __name__ == '__main__':
    budget = float(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    log = open(os.path.join(HERE, 'runs.jsonl'), 'a')
    t0 = time.time(); n = 0
    with Pool(workers, initializer=init) as P:
        for r in P.imap_unordered(trial, range(start, start + 10 ** 7), chunksize=1):
            log.write(json.dumps(r) + '\n'); log.flush(); n += 1
            if time.time() - t0 > budget:
                P.terminate(); break
    print('trials', n)
