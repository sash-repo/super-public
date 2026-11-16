"""The census: every configuration of three 2-tensor fibers over a periplectic
form (and, for contrast, over an even one).

    python3 runfib.py <form> <k> <N> [nsamp]     one order, all eight types
    python3 runfib.py batch                      the whole table

Prints one line per configuration as it is finished.
"""

import itertools
import os
import sys
import time

from sfib import MSetup
from srunfib import analyse

HERE = os.path.dirname(os.path.abspath(__file__))
TYPE = {0: "S^2", 1: "Lam^2"}


def one(form, k, N, shifts, nsamp=2):
    n = k if form == "odd" else 0
    m = k
    S = MSetup(m, n, shifts=shifts, form=form)
    t0 = time.time()
    schemes, invs, ns, dim = analyse(S, N, nsamp=nsamp, verbose=False)
    line = (f"{form} k={k} N={N} "
            f"{TYPE[shifts[0]]:5s} x {TYPE[shifts[1]]:5s} -> {TYPE[shifts[2]]:5s}: "
            f"dim {dim:2d}   ({len(schemes)} schemes, {len(invs)} invariant "
            f"element(s) in the target, kernel {len(ns)}, {time.time()-t0:.0f}s)")
    print(line, flush=True)
    return dim


def sweep(form, k, N, nsamp=2):
    for shifts in itertools.product((0, 1), repeat=3):
        one(form, k, N, shifts, nsamp=nsamp)


def main():
    if sys.argv[1] == "batch":
        for form, k, N in [("odd", 2, 0), ("odd", 2, 2), ("odd", 3, 0),
                           ("odd", 3, 2), ("even", 1, 0), ("even", 1, 2),
                           ("odd", 2, 4)]:
            sweep(form, k, N)
    else:
        form, k, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        nsamp = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        sweep(form, k, N, nsamp)


if __name__ == "__main__":
    main()
