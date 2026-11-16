"""The same unary census over an EVEN symplectic form -- the contrast.

Over an even form omega^{ij} is antisymmetric and a metric A_{ij} super-symmetric,
so the trace omega^{ij}A_{ij} vanishes identically and every composite built from
it dies with it.  This script checks that, and asks what unary invariants are left,
so that the statement "Grozman's Gz is not a composition" can be made by counting
rather than by inspection.

Usage:  python3 sevencensus.py [m] [n]
"""

import random
import sys
from fractions import Fraction

from ssuper import SSetup
from sunary import invariant_unary
from sodd0 import ptr


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    S = SSetup(m, n, form="even")
    A = S.A
    print(f"EVEN form, (m,n) = ({m},{n}): sdim of the manifold = ({2*m}|{n}), "
          f"h({2*m}|{n}), aut(omega) = osp({n}|{2*m})")

    rng = random.Random(2718 + 10 * m + n)
    zero = True
    for _ in range(4):
        P = S.random_field(3, rng, nterms=3)
        if ptr(S, P):
            zero = False
    print(f"the trace omega^{{ij}}A_{{ij}} is identically zero: "
          f"{'YES' if zero else 'NO'}")

    print("unary invariant differential operators on metrics:")
    for order in (0, 1, 2):
        for degree in (2, 1, 0):
            rng = random.Random(3000 + 100 * order + 10 * degree + m)
            invariant_unary(S, order, degree, rng)


if __name__ == "__main__":
    main()
