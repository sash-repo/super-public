#!/usr/bin/env python3
"""Where can the multiplicity-freeness (118) fail?  The count of lowest-weight
vectors in SM_{mu1} (x) SM_{mu2} is weight-INDEPENDENT (tensor.py), so any
failure must come from the degeneracy of the modules themselves.  This scans
the special weights: the dimension of the space of invariant bilinear
differential operators of order <= N, for weight pairs with mu in (1/2)Zee.
"""
import sys
from fractions import Fraction as F
import fullops as FO

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
half = [F(n, 2) for n in range(-6, 7)]
gen = (F(1, 31), F(5, 31))
print(f"# order <= {N};  entry = dim Hom for the shift s = 0,1,2,...,{2*N}")
print(f"# generic reference pair {gen}")
ref = [len(FO.solve(N, gen[0], gen[1], F(s))[1]) for s in range(0, 2*N+1)]
print(f"{'mu1':>6} {'mu2':>6}   dims per shift s=0..{2*N}")
print(f"{'gen':>6} {'gen':>6}   {ref}")
odd = 0
for m1 in half:
    for m2 in half:
        d = [len(FO.solve(N, m1, m2, F(s))[1]) for s in range(0, 2*N+1)]
        if d != ref:
            odd += 1
            print(f"{str(m1):>6} {str(m2):>6}   {d}   <-- differs")
print(f"\n# weight pairs differing from the generic answer: {odd} "
      f"of {len(half)**2}")
