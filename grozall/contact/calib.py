"""Calibration of genn.py (any n) against the answers of the dimension-3 engine.

The scheme ansatz must reproduce, for n = 1, every census of the dimension-3 engine:
densities (1 at W=0, 1 at W=2, 0 otherwise), metrics (three contractions at W=0, and
one operator at W=2 exactly on lambda+mu=0, into F_3 alone).
"""

import sys
from fractions import Fraction as Fr

from genn import Problem, schemes, scheme_name, target_weight

CASES = []


def run(n, k1, l1, k2, l2, k3, W, want, note=""):
    P = Problem(n, k1, k2, k3, W)
    if not P.schemes:
        d = 0
    else:
        l3, ns = P.solve(l1, l2)
        d = P.op_dim(ns)
    ok = (d == want)
    print("  %-5s n=%d T(%d,%s)xT(%d,%s)->T(%d,.) W=%d : ansatz %2d, dim %d (want %d) %s"
          % ("ok" if ok else "FAIL", n, k1, l1, k2, l2, k3, W, P.dim_ansatz(), d, want,
             note))
    CASES.append(ok)
    return d


def main():
    print("k(3), the densities :")
    for W in (0, 1, 2, 3, 4):
        want = 1 if W in (0, 2) else 0
        for (l, m) in [(Fr(0), Fr(0)), (Fr(2, 3), Fr(-1, 5)), (Fr(-1), Fr(-1))]:
            run(1, 0, l, 0, m, 0, W, want)

    print("\nk(3), the metrics :")
    for k3 in (0, 2, 4):
        run(1, 2, Fr(1, 3), 2, Fr(2, 5), k3, 0, 1)
    # W = 2: 1 on lambda+mu=0 for k3=0, else 0
    run(1, 2, Fr(1, 3), 2, Fr(-1, 3), 0, 2, 1, "(on the line)")
    run(1, 2, Fr(1, 3), 2, Fr(2, 5), 0, 2, 0, "(off the line)")
    run(1, 2, Fr(0), 2, Fr(0), 0, 2, 1, "(lambda=mu=0)")
    run(1, 2, Fr(1, 3), 2, Fr(-1, 3), 2, 2, 0)
    run(1, 2, Fr(1, 3), 2, Fr(-1, 3), 4, 2, 0)

    print("\n%d of %d calibration cases passed" % (sum(CASES), len(CASES)))
    return 0 if all(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
