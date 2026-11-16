"""The graph census at size 8 (Statement 8.6.2, the row n = 8, and table (62)).

The matrices of size 8 are read from `level8.pkl`, built by `par8.py`,
and by Proposition 8.6.1 the collapse to graphs costs nothing but a canonical
form of the unsigned graph, so the row is within reach.

The lattice is constant along a graph -- matrices with one graph are one orbit under
the sign changes, which are isometries -- so `root_system_type` is called once per
graph and not once per matrix.  The per-lattice matrix counts that come out of it are
compared with the row n = 8 of table (58) (A_8 2891, D_8 11657, E_8 121068),
which `lat8.py` computes.

For the Carter matrices the classes of w = w_1 w_2 are computed as well; that is what
carries Statement 8.6.3 to n = 8, where E_8 enters and, by Corollary 8.5.3,
the E column appears for the last time.

Run:  python3 graphs8.py
"""

import collections
import multiprocessing as mp
import pickle
import sys
import time

import vinb
from carter import graph_canon, weyl_element, order, charpoly, _fmt
from lattice import root_system_type


def _chunk(mats):
    """graph -> (number of matrices with that graph, one of them)."""
    out = {}
    for A in mats:
        g = graph_canon(A)
        if g in out:
            out[g][0] += 1
        else:
            out[g] = [1, A]
    return out


def main():
    t0 = time.time()
    with open('level8.pkl', 'rb') as f:
        mats = pickle.load(f)
    conn = [A for A in mats if vinb.is_connected(A)]
    print('level8.pkl: %d matrices, %d of them indecomposable  (%.0f s)'
          % (len(mats), len(conn), time.time() - t0), flush=True)

    nproc = 8
    chunks = [conn[i::nproc] for i in range(nproc)]
    with mp.Pool(nproc) as pool:
        parts = pool.map(_chunk, chunks)
    merged = {}
    for p in parts:
        for g, (cnt, A) in p.items():
            if g in merged:
                merged[g][0] += cnt
            else:
                merged[g] = [cnt, A]
    print('graphs: %d   (%.0f s)' % (len(merged), time.time() - t0), flush=True)

    # the lattice, once per graph
    info = {}
    for g, (cnt, A) in merged.items():
        info[g] = (cnt, root_system_type(A)[0], vinb.is_bipartite(A), A)
    print('lattices identified   (%.0f s)' % (time.time() - t0), flush=True)

    mat_by_lat, gra_by_lat = collections.Counter(), collections.Counter()
    cmat_by_lat, cgra_by_lat = collections.Counter(), collections.Counter()
    for g, (cnt, lat, bip, A) in info.items():
        mat_by_lat[lat] += cnt
        gra_by_lat[lat] += 1
        if bip:
            cmat_by_lat[lat] += cnt
            cgra_by_lat[lat] += 1

    print('\ntype Pos, n=8:')
    for lat in sorted(mat_by_lat):
        print('   %-4s  matrices %7d   graphs %4d' % (lat, mat_by_lat[lat], gra_by_lat[lat]))
    print('   %-4s  matrices %7d   graphs %4d'
          % ('all', sum(mat_by_lat.values()), len(info)))
    known = {'A8': 2891, 'D8': 11657, 'E8': 121068}
    ok = all(mat_by_lat[k] == v for k, v in known.items())
    print('   agrees with the census table row (A_8 2891, D_8 11657, E_8 121068): %s'
          % ('yes' if ok else 'NO'))

    print('\nCarter matrices, n=8:')
    for lat in sorted(cmat_by_lat):
        print('   %-4s  matrices %7d   graphs %4d'
              % (lat, cmat_by_lat[lat], cgra_by_lat[lat]))
    print('   %-4s  matrices %7d   graphs %4d'
          % ('all', sum(cmat_by_lat.values()), sum(cgra_by_lat.values())))
    print('   the census table gives 72, 268, 823, total 1163: %s'
          % ('yes' if (cmat_by_lat['A8'], cmat_by_lat['D8'], cmat_by_lat['E8'])
             == (72, 268, 823) else 'NO'))

    print('\nthe elements w = w_1 w_2 of the Carter graphs of size 8:')
    rows = collections.defaultdict(list)
    for g, (cnt, lat, bip, A) in info.items():
        if bip:
            w = weyl_element(A)
            rows[lat].append((order(w), tuple(charpoly(w)), cnt))
    for lat in sorted(rows):
        rows[lat].sort()
        print('   %s: %d graphs, orders %s'
              % (lat, len(rows[lat]), ', '.join(str(o) for o, _, _ in rows[lat])))
        seen = collections.defaultdict(list)
        for o, cp, cnt in rows[lat]:
            seen[cp].append(o)
        print('        distinct characteristic polynomials: %d' % len(seen))
        for cp in sorted(seen, key=lambda c: (seen[c][0], c)):
            print('        %-44s order %-3s (%d graph%s)'
                  % (_fmt(list(cp)), seen[cp][0], len(seen[cp]),
                     '' if len(seen[cp]) == 1 else 's'))

    ell = all(sum(cp) != 0 for lat in rows for _, cp, _ in rows[lat])
    print('\nevery w is elliptic (no eigenvalue 1): %s' % ('yes' if ell else 'NO'))

    with open('graphs8.pkl', 'wb') as f:
        pickle.dump({g: (v[0], v[1], v[2], v[3]) for g, v in info.items()}, f)
    print('total %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
