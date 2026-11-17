"""The census with one odd parameter: every family over every retract (a,b),
a <= 6, tested for non-splitness and for homogeneity.

For each class the orbit dimension in P(H^1(T_1)) is computed too, so that the
families can be grouped by orbit: a stratum where sdim_C v and the homogeneity
are constant and the orbit is of dimension dim P(H^1) is the whole of a dense
orbit, and the classification is then complete on that stratum.
"""
from fractions import Fraction
from itertools import combinations
import oddpar as O, famtools as F
from ns import SM

DEG = 9


def classes(n):
    """The basis vectors, their pairwise and triple sums, one generic class."""
    out = []
    for i in range(n):
        out.append([1 if j == i else 0 for j in range(n)])
    for i, j in combinations(range(n), 2):
        out.append([1 if k in (i, j) else 0 for k in (range(n))])
        out.append([1 if k == i else (2 if k == j else 0) for k in range(n)])
    if n >= 3:
        out.append([1] * n)
    pr = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    out.append(pr[:n])
    return out


print(__doc__, flush=True)
import sys
AMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
AMIN = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for a in range(AMIN, AMAX + 1):
    for b in range(0, a + 1):
        ks = [a, b]
        L = O.Layer(ks, 1)
        if L.dim == 0:
            continue
        sp = SM.split(ks, npar=1)
        d0, t0 = sp.sdim_vect(DEG), sp.transitive(DEG)
        print(f"\n=== retract {tuple(ks)}:  dim H^1(T_1) = {L.dim},"
              f"  P^{L.dim-1};  split family sdim_C v = {d0}, homog {t0}")
        seen = {}
        for v in classes(L.dim):
            vv = [Fraction(c) for c in v]
            od = L.orbit_dim_proj(vv)
            M = F.cocycle_to_family(ks, F.vec_to_coc(L.Q, vv))
            d = M.sdim_vect(DEG)
            h = M.transitive(DEG)
            key = (od, d, h)
            seen.setdefault(key, []).append(v)
        for (od, d, h), vs in sorted(seen.items()):
            tag = "  <-- dense orbit" if od == L.dim - 1 else ""
            print(f"   orbit dim {od:>2}  sdim_C v = {str(d):>9}  homogeneous"
                  f" {str(h):>5}  ({len(vs)} classes){tag}")
            print(f"       e.g. {F.show(L.Q, [Fraction(c) for c in vs[0]])}",
                  flush=True)
