#!/usr/bin/env python3
"""Statement 5.13: the derivations of svect(1|2;1)^(1) and svect(1|1;2)^(1)
(p = 3, 5) and of h(0|4)^(1) induced by no element of Der O.

For each algebra prints dim Der L, dim N_{Der O}(L), dim C_{Der O}(L), the gap,
and for each (degree, parity) block with a gap a representative derivation D,
by its values on a basis of L (x0 = x, y0, y1 = xi_1, xi_2; d0 = d/dx, e_j =
d/dxi_j; x0^(k) is the divided power).  Then dim Der L is recomputed with no
grading used (brute.py)."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
from dervect import Super, Echelon, svect, derived, poisson  # noqa: E402
from extra import report                                     # noqa: E402
from brute import der_dim_brute                              # noqa: E402


def basis(S, L):
    return [S.unvec(r) for r in Echelon([S.vec(X) for X in L], S.p).rows]


def main():
    for p in (3, 5):
        for (m, n, N) in [(1, 2, [1]), (1, 1, [2])]:
            S = Super(p, m, n, N)
            L = basis(S, derived(S, svect(S)))
            tot, _, _ = report(f"p={p} svect({m}|{n};{N})^(1)", S, L)
            print(f"    ungraded recomputation: dim Der L = "
                  f"{der_dim_brute(S, L)} (graded: {tot})", flush=True)
        S = Super(p, 0, 4)
        L, _ = poisson(S, 'h', 0)
        L = basis(S, derived(S, L))
        tot, _, _ = report(f"p={p} h(0|4)^(1)", S, L)
        print(f"    ungraded recomputation: dim Der L = "
              f"{der_dim_brute(S, L)} (graded: {tot})", flush=True)


if __name__ == "__main__":
    main()
