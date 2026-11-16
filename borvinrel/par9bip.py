"""The no-odd-cycle (Carter) branch of the census, carried to n = 9, in parallel.

Having no odd cycle is inherited by principal submatrices, so it may be imposed at
every level without losing anything: a matrix of size n with no odd cycle can only
come from a parent of size n-1 with no odd cycle.  That keeps the branch small
enough to reach n = 9, where the full census is already out of reach.
"""

import multiprocessing as mp
import sys
import time

import permcensus as pc
import vinb


def work(chunk):
    out = {}
    for A in chunk:
        for B in vinb.extend(A):
            if not vinb.is_bipartite(B):
                continue
            c = pc.perm_canon(B)
            if c not in out:
                out[c] = B
    return out


def level(parents, nproc=8):
    chunks = [parents[i::nproc] for i in range(nproc)]
    with mp.Pool(nproc) as pool:
        parts = pool.map(work, chunks)
    merged = {}
    for p in parts:
        merged.update(p)
    return list(merged.values())


def report(n, mats):
    conn = [B for B in mats if vinb.is_connected(B)]
    cdyn = [B for B in conn
            if all(B[i][j] <= 0 for i in range(n) for j in range(n) if i != j)]
    print(f'n={n:2d}  no-odd-cycle: all={len(mats):7d}  connected={len(conn):7d}'
          f'   without the {len(cdyn)} simply-laced Cartan '
          f'matrices): {len(conn) - len(cdyn):7d}', flush=True)


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    t0 = time.time()
    cur = [((2,),)]
    for n in range(2, nmax + 1):
        cur = level(cur) if n >= 7 else [B for A in cur for B in vinb.extend(A)
                                         if vinb.is_bipartite(B)]
        if n < 7:
            seen = {}
            for B in cur:
                c = pc.perm_canon(B)
                seen.setdefault(c, B)
            cur = list(seen.values())
        report(n, cur)
        print(f'    ({time.time() - t0:.0f} s)', flush=True)
        import pickle
        with open(f'bip{n}.pkl', 'wb') as f:
            pickle.dump(cur, f)


if __name__ == '__main__':
    main()
