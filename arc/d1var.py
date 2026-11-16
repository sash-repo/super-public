#!/usr/bin/env python3
"""Independent check: the uniform formula beta^N o Delta^* o Bhat_N is invariant
in the convention of the text,  Q(B(f,g)) = B(Qf,g) + (-1)^{p(f)} B(f,Qg),
and Delta^* o Bhat_N alone is NOT (it is invariant in the other convention).

Run:  ../venv/bin/python d1var.py [NMAX]
"""
import sys
import dproof as P

if __name__ == "__main__":
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for N in range(NMAX + 1):
        a = P.check_invariance_1var(N, twist=True)
        b = P.check_invariance_1var(N, twist=False)
        print("N=%d  with beta^N: %s | without: %s"
              % (N, "invariant" if not a else "NO",
                 "invariant" if not b else "NO"))
