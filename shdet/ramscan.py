"""Scan the four sign conventions of the Ramond bracket: which one makes both
the Jacobi identity and the table (74) hold."""
from fractions import Fraction as Q
from itertools import product as iproduct
import sys
from kM4 import monomials, jacobi_bad, check_cocycle, name_of

B = int(sys.argv[1]) if len(sys.argv) > 1 else 2
small = monomials(range(-B, B + 1))
jac = monomials(range(0, 2))          # cheap Jacobi probe
for sigma in (1, -1):
    for k in (Q(1), Q(-1)):
        for kz in (Q(1), Q(-1)):
            kw = dict(sigma=sigma, k=k, kz=kz)
            j = jacobi_bad(jac, **kw)
            f = check_cocycle(small, **kw)
            tot = sum(len(v) for v in f.values())
            print("sigma=%2d k=%2s kz=%2s : Jacobi bad %5d, cocycle fails %5d"
                  % (sigma, k, kz, j, tot), flush=True)
