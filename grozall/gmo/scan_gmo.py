"""Where does dim End jump?  The scan that settles [GMO]'s 4/5."""
import sys
from fractions import Fraction as Q
from gmo import dim_symmetry, scan

def line(k, fix, value, cands, base):
    print(f"  k={k}, {fix}={value}: weights where dim > {base}")
    for (lam, mu), d in scan(k, fix, value, cands):
        if d > base:
            print(f"      (lam,mu)=({lam},{mu})  dim={d}")

# a dense grid of small rationals
cands = sorted({Q(p, q) for q in (1, 2, 3, 4, 5, 6, 8) for p in range(-30, 31)})

print("=== k=4 ===")
line(4, "lam", Q(0), cands, 2)
line(4, "mu", Q(1), cands, 2)
print("=== the two candidates, with a larger order bound ===")
for lam, mu in ((Q(0), Q(4,5)), (Q(0), Q(5,4)), (Q(1,5), Q(1)), (Q(-1,4), Q(1))):
    for rmax in (4, 6):
        print(f"    k=4 (lam,mu)=({lam},{mu}) rmax={rmax}: dim={dim_symmetry(4, lam, mu, rmax=rmax)}")
print("=== k=3 ===")
line(3, "lam", Q(0), cands, 1)
print("=== k=5 ===")
line(5, "lam", Q(0), cands, 1)
