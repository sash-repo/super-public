"""Lattice split of the partial Cartan matrices of size n (no odd cycle, connected),
read from bip{n}.pkl, which `par9bip.py` writes (the rows n = 9, 10 of table (58)).

Usage:  python latbip.py 9      (after `python par9bip.py 10`)
"""
import collections
import pickle
import sys

import vinb
import lattice

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    with open(f'bip{n}.pkl', 'rb') as f:
        mats = pickle.load(f)
    conn = [B for B in mats if vinb.is_connected(B)]
    cnt = collections.Counter(lattice.root_system_type(B)[0] for B in conn)
    print(f'n={n}: {len(conn)} connected matrices with no odd cycle; lattice split:',
          dict(sorted(cnt.items())), ' total', sum(cnt.values()))
