#!/usr/bin/env python3
"""k(1|5): the exceptional weights of the binary problem.

For each order N, the dimension of the space of i(osp(5|2))-invariant operators
F_lam1 (x) F_lam2 --> F_{lam1+lam2+N} is computed over a grid of weights; the
generic value is the most frequent one, and the pairs where the dimension jumps
are printed.  Same economy as in n5bin.py (the torus filter), same caveat: a
jump is a lower bound on nothing -- it is an exact dimension at that pair.

Usage:  n5exc.py N1,N2,... [halfstep] [range]
"""
import sys
from fractions import Fraction as Fr

from n5bin import Bn, gens_osp
from n5 import ALG
from knclass import label


def scan(N, grid, tmax=2):
    dims = {}
    for i, l1 in enumerate(grid):
        for l2 in grid[i:]:
            C = Bn(gens_osp(), l1, l2)
            dims[(l1, l2)] = len(C.solve(N, tmax=tmax)[1])
    vals = list(dims.values())
    base = max(set(vals), key=vals.count)
    print(f"order {N}: generic dim = {base}; "
          f"{sum(1 for v in vals if v != base)} weight pairs with a different "
          f"dim (grid of {len(grid)} values)")
    for k, v in sorted(dims.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if v != base:
            print(f"    lam1={k[0]}, lam2={k[1]}: dim = {v}")
    sys.stdout.flush()
    return base, dims


if __name__ == '__main__':
    Ns = [int(x) for x in sys.argv[1].split(',')]
    step = Fr(sys.argv[2]) if len(sys.argv) > 2 else Fr(1)
    lo, hi = (-8, 4) if len(sys.argv) <= 3 else \
        tuple(int(x) for x in sys.argv[3].split(','))
    grid = []
    x = Fr(lo)
    while x <= hi:
        grid.append(x)
        x += step
    grid.append(Fr(3, 7))                     # one generic value for control
    for N in Ns:
        scan(N, grid)
