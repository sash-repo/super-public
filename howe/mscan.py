"""
mscan.py -- box scan: all singular vectors of degree <= mmax in I(V(lambda) (x) C_c)
over M(n) (or over M_{a,b}(n)) for lambda in a dominant box and c in a range.

usage:  python3 mscan.py <n> <mmax> <lo> <hi> <clo> <chi> [a b]
"""

import sys
from fractions import Fraction as F
from itertools import product

from mn import layer, mbr, mdeg
from mimod import Mn, Mab, MInduced
from mone import scan


def dominant(n, lo, hi):
    for lam in product(range(hi, lo - 1, -1), repeat=n):
        if all(lam[i] >= lam[i + 1] for i in range(n - 1)):
            yield lam


def ops_for(mn, n, a, b):
    A = Mab(mn, a, b)
    ops = list(A.layer(1))
    from mgencheck import span_dim
    amb = [next(iter(f)) for f in layer(n, 2)]
    gen = []
    for f in A.layer(1):
        for g in A.layer(1):
            x = mbr(f, g, n)
            if x and mdeg(x, n) == 2:
                gen.append(x)
    if span_dim(gen, amb) < len(A.layer(2)):
        ops += A.layer(2)
    return ops


if __name__ == '__main__':
    n = int(sys.argv[1])
    mmax = int(sys.argv[2])
    lo, hi = int(sys.argv[3]), int(sys.argv[4])
    clo, chi = int(sys.argv[5]), int(sys.argv[6])
    ab = (F(sys.argv[7]), F(sys.argv[8])) if len(sys.argv) > 8 else None
    mn = Mn(n)
    ops = ops_for(mn, n, *ab) if ab else None
    tag = "M_{%s,%s}(%d)" % (ab[0], ab[1], n) if ab else "M(%d)" % n
    for lam in dominant(n, lo, hi):
        for c in range(clo, chi + 1):
            scan(n, lam, c, mmax, ops, tag)
            sys.stdout.flush()
