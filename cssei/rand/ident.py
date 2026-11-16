#!/usr/bin/env python3
"""Identify the simple factors found by search.py.
dim <= 31, centroid F_2: CSS invariants (#2-nilpotent, #toral, both incl. 0) against the
CSS library (all its algebras, computed here) -> css_inv.json.
dim > 31: fingerprint (dim, centroid, restricted, generic rank of ad) against the
simple parts of the ambients and of psl(n)."""
import sys, os, json, glob, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cmp'))
from count import count
from lie import factors, centroid_dim, restricted
from loadcss import load, library
from gf2 import rank

def masks(ad):
    d = ad.shape[0]
    w = (1 << np.arange(d, dtype=np.uint64)).astype(np.uint64)
    return np.array([[int((ad[i, :, j].astype(np.uint64) * w).sum()) for j in range(d)] for i in range(d)], np.uint32)

def inv(ad):
    d = ad.shape[0]
    return tuple(int(v) for v in count(masks(ad), d, min(d, 16)))

def genrank(ad, rng, n=64):
    d = ad.shape[0]
    return max(rank((ad[np.nonzero(rng.integers(0, 2, d))[0]].sum(0) & 1).astype(np.uint8)) for _ in range(n))

if __name__ == '__main__':
    cache = 'css_inv.json'
    if os.path.exists(cache):
        css = json.load(open(cache))
    else:
        css = {}
        for nm, p in library():
            ad = load(p)
            css[nm] = [ad.shape[0], *inv(ad)]
            print(nm, css[nm], flush=True)
        json.dump(css, open(cache, 'w'), indent=0)
    bykey = collections.defaultdict(list)
    for nm, (d, a, t) in css.items():
        bykey[(d, a, t)].append(nm)
    # counts per fingerprint
    cnt = collections.Counter(); amb = collections.defaultdict(collections.Counter)
    if os.path.exists('runs.jsonl'):          # the log of search.py, if it has been run here
        for l in open('runs.jsonl'):
            r = json.loads(l)
            for f in r['factors']:
                cnt[f['fp']] += 1; amb[f['fp']][r['amb']] += 1
    rows = []
    for p in sorted(glob.glob('found/*.npz')):
        z = np.load(p); fp = str(z['fp']); ad = z['ad']; d = ad.shape[0]
        d_, c, r, st, mx = fp.split(':')
        rec = {'fp': fp, 'file': os.path.basename(p), 'n': cnt[fp], 'amb': dict(amb[fp])}
        if st == 'simple' and c == '1' and d <= 31:
            a, t = inv(ad)
            rec['inv'] = [a, t]
            rec['css'] = bykey.get((d, a, t), [])
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(rows, open('ident.json', 'w'), indent=1)
