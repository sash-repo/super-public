#!/usr/bin/env python3
"""The trace forms omega_k = tr a_k on f4 (26-dimensional module) and on e6
(27-dimensional module), k odd, 3 <= k <= top.

The models are those of runexc.py (built in albert.py); a_k is evaluated on
random elements by the subset recursion of expo.py, modulo p = 65521.  A nonzero
trace modulo p is a nonzero trace over Q; a zero is the verdict of `trials`
independent random tests.
"""

import sys
import numpy as np

from expo import a_k_dp
from runexc import f4_basis, e6_basis, PP


def trsurvey(name, basis, ks, trials=3, seed=11, p=PP):
    N = basis[0].shape[0]
    B = [np.asarray(b, dtype=np.int64) % p for b in basis]
    rng = np.random.default_rng(seed)
    good = []
    for k in ks:
        nz = False
        for _ in range(trials):
            c = rng.integers(0, p, size=(k, len(B)))
            mats = []
            for t in range(k):
                M = np.zeros((N, N), dtype=np.int64)
                for s in range(len(B)):
                    M = (M + int(c[t, s]) * B[s]) % p
                mats.append(M)
            A = a_k_dp(mats, N, p=p) % p
            if int(np.trace(A)) % p:
                nz = True
                break
        print(f"  {name} tr a_{k}: {'nonzero' if nz else 'zero'}")
        sys.stdout.flush()
        if nz:
            good.append(k)
    print(f"{name}: tr a_k =/= 0 for k in {good}")


if __name__ == "__main__":
    top = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    if which in ("both", "f4"):
        trsurvey("f4", f4_basis(), range(3, top + 1, 2))
    if which in ("both", "e6"):
        trsurvey("e6", e6_basis(), range(3, top + 1, 2))
