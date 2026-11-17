#!/usr/bin/env python3
"""The bridge between Theorem 1.6 and the exponents: for which k is the invariant
k-form tr a_k(X_1,...,X_k) not identically zero?

These forms are the classical transgression (primitive) cocycles of g: the class
of tr a_{2m+1} generates H^{2m+1}(g;K) when m is an exponent whose basic invariant
is a power trace in the representation used.  The point of the test is that
tr a_{k+1} =/= 0 forces a_k =/= 0, which is what puts the doubled exponents into
A(g).
"""

import sys
import numpy as np

from expo import a_k_dp, P
from runexpo import CASES, EXPONENTS
from expo import rand_elt


def trsurvey(name, basis, ks, trials=3, seed=11):
    N = basis[0].shape[0]
    B = [np.asarray(b, dtype=np.int64) % P for b in basis]
    rng = np.random.default_rng(seed)
    good = []
    for k in ks:
        nz = False
        for _ in range(trials):
            mats = [rand_elt(B, rng) for _ in range(k)]
            A = a_k_dp(mats, N) % P
            if int(np.trace(A)) % P:
                nz = True
                break
        if nz:
            good.append(k)
    print(f"  {name:8s} tr a_k =/= 0 for k in {good}")
    ex = EXPONENTS.get(name)
    if ex:
        print(f"  {name:8s} 2m+1 over the exponents m: "
              f"{sorted(set(2 * m + 1 for m in ex))}")
    sys.stdout.flush()


if __name__ == "__main__":
    want = sys.argv[1:] or [c[0] for c in CASES]
    for name, basis, ks in CASES:
        if name in want:
            trsurvey(name, basis, range(3, max(ks) + 1, 2))
