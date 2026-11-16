"""One more term of the census: n = 8, in parallel over the cores of this machine.

The level-7 representatives are split between the workers; each worker extends its
share and returns the set of canonical forms it met, and the sets are united at the
end.  That is sound because the construction is complete parent by parent: every
orbit at level 8 is reached from EVERY level-7 orbit sitting under it, so no orbit
can be lost by cutting the parents into blocks.
"""

import multiprocessing as mp
import pickle
import sys
import time

import permcensus as pc
import vinb


def work(chunk):
    out = {}
    for A in chunk:
        for B in vinb.extend(A):
            c = pc.perm_canon(B)
            if c not in out:
                out[c] = B
    return out


def main():
    t0 = time.time()
    reps, _ = pc.census(7, verbose=False)
    parents = reps[7]
    print(f'level 7: {len(parents)} representatives', flush=True)
    nproc = 8
    chunks = [parents[i::nproc] for i in range(nproc)]
    with mp.Pool(nproc) as pool:
        parts = pool.map(work, chunks)
    merged = {}
    for p in parts:
        merged.update(p)
    print(f'level 8: {len(merged)} representatives  ({time.time() - t0:.0f} s)', flush=True)
    mats = list(merged.values())
    conn = [B for B in mats if vinb.is_connected(B)]
    bip = [B for B in mats if vinb.is_bipartite(B)]
    cbip = [B for B in mats if vinb.is_connected(B) and vinb.is_bipartite(B)]
    dyn = [B for B in conn if all(B[i][j] <= 0 for i in range(8) for j in range(8) if i != j)]
    cdyn = [B for B in cbip if all(B[i][j] <= 0 for i in range(8) for j in range(8) if i != j)]
    print(f'n=8  all={len(mats)}  connected={len(conn)}  no-odd-cycle={len(bip)}  '
          f'connected & no-odd-cycle={len(cbip)}')
    print(f'     simply-laced Cartan matrices among the connected: {len(dyn)} '
          f'(dets {sorted(vinb.det_int(B) for B in dyn)})')
    print(f'     without the simply-laced Cartan matrices: connected {len(conn) - len(dyn)}, '
          f'connected & no odd cycle {len(cbip) - len(cdyn)}')
    with open('level8.pkl', 'wb') as f:
        pickle.dump(mats, f)


if __name__ == '__main__':
    main()
