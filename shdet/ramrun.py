"""
The Ramond cocycle (74) at the convention of the paper's own eq. (kb):
  [f,g] = (2-E)(f) D(g) - D(f) (2-E)(g)
          - (-1)^{p(f)} ( d_xi f d_eta g + d_eta f d_xi g + d_theta f d_theta g
                          + (1/t) d_zeta f d_zeta g ),
i.e. kM4's (sigma, k, kz) = (-1, 1, 1).
"""
from fractions import Fraction as Q
import sys
from kM4 import monomials, jacobi_bad, check_skew, check_cocycle, name_of

CONV = dict(sigma=-1, k=Q(1), kz=Q(1))
B = int(sys.argv[1]) if len(sys.argv) > 1 else 3

print("Jacobi for the Ramond bracket of eq. (kb) on |a| <= 1:",
      "OK" if jacobi_bad(monomials(range(-1, 2)), **CONV) == 0 else "FAILS")
small = monomials(range(-B, B + 1))
bad = check_skew(small)
print("super-skew-symmetry of (74):",
      "OK" if not bad else "%d violations" % len(bad))
f = check_cocycle(small, **CONV)
tot = sum(len(v) for v in f.values())
print("cocycle identity, window |a| <= %d (%d monomials): %d failing triples"
      % (B, len(small), tot))
for sig in sorted(f, key=lambda s: -len(f[s]))[:8]:
    ex = f[sig][0]
    print("   %-24s %5d  e.g. %s %s %s -> %s"
          % ("/".join(sig), len(f[sig]), name_of(ex[0])[:2],
             name_of(ex[1])[:2], name_of(ex[2])[:2], ex[3]))
