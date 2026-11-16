#!/usr/bin/env python3
"""Regenerate the eleven simple constituents of dim <= 31 of the random search whose invariants
are in no entry of [CSS, Table C.3], from one seed each, without the logs of the search.

A trial of rand/search.py (seeds < 2000000) or rand/search2.py (seeds >= 2000000) depends on its
seed only; the trial is replayed, the composition factors of the generated subalgebra are found,
and the simple factor with the given (dim, #2-nilpotent, #toral) is kept (invariants counted
with zero included, as in [CSS]).  For each algebra: the seed, the ambient, the dimension of the
centroid (1 = absolutely simple), restrictedness; the table is written to OUTDIR/k<d>_<nil>_<tor>.npz
(ad[i][k][j] = coefficient of e_k in [e_i,e_j]).

usage: regen.py [OUTDIR]        (default: regen_out/ in this folder)"""
import os, sys
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recheck as R
from lie import closure, table_of, factors, centroid_dim, restricted
from count import invariants

# (dim, #2-nilpotent, #toral): the first seed that produced it
SEEDS = {
    (15, 112, 41): 1001651, (15, 176, 65): 1097506, (28, 27584, 12545): 2000165,
    (30, 42752, 12545): 2000039, (30, 46080, 28673): 1465481, (30, 67328, 37121): 2006681,
    (31, 63488, 10241): 1011741, (31, 92416, 99585): 2021746, (31, 133120, 12289): 1116925,
    (31, 143360, 10241): 1369063, (31, 159744, 16385): 1102656,
}

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'regen_out')
    os.makedirs(out, exist_ok=True)
    R.init()
    import ambients as A
    for t, seed in SEEDS.items():
        rng = np.random.default_rng(seed)
        if seed < 2000000:
            a = R.AMS1[seed % len(R.AMS1)]
            mode = 'graded' if (a.deg is not None and rng.random() < 0.5) else 'sparse'
            if mode == 'graded':
                gens = A.graded_pair(a, rng, kmax=int(rng.integers(1, 5)))
            else:
                g = 2 if rng.random() < 0.8 else 3; kmax = int(rng.choice([1, 2, 3, 4, 6]))
                gens = [A.sparse(a, rng, kmax) for _ in range(g)]
        else:
            a = R.AMS2[seed % len(R.AMS2)]; mode, gens = R.S2.gens_for(a, rng)
        E = closure(a, gens, a.L); ad = table_of(a, E); res = []; factors(ad, rng, res)
        hit = None
        for fad, st, AK in res:
            if st == 'simple' and fad.shape[0] == t[0]:
                if tuple(int(v) for v in invariants(R.tabify(fad))) == t[1:]:
                    hit = (fad, AK)
        if hit is None:
            print(t, 'NOT REPRODUCED from seed', seed, flush=True); continue
        fad, AK = hit
        np.savez(os.path.join(out, 'k%d_%d_%d.npz' % t), ad=fad)
        print(t, 'seed', seed, 'ambient', a.name, 'centroid dim', centroid_dim(fad, AK),
              'restricted', restricted(fad), flush=True)
