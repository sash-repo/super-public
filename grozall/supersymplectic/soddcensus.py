"""The census of unary invariant operators on metrics over an ODD symplectic form,
for the orders and targets a composition of total order 2 can use.

Usage:  python3 soddcensus.py [k]
"""

import random
import sys

from ssuper import SSetup
from sunary import invariant_unary, enumerate_unary, evaluate_unary


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    S = SSetup(k, k, form="odd")
    print(f"PERIPLECTIC k = {k}: sdim ({k}|{k}), le({k}), aut(omega) = pe({k})")
    print("unary invariant differential operators on metrics:")
    for order in (0, 1, 2):
        for degree in (2, 1, 0):
            rng = random.Random(9000 + 100 * order + 10 * degree + k)
            invariant_unary(S, order, degree, rng)
    print("\n(a composition of total order 2 pairs P(A) and Q(B) with"
          " ord P + ord Q = 2)")


if __name__ == "__main__":
    main()
