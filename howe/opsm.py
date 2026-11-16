#!/usr/bin/env python3
r"""opsm.py -- the singular vectors over m(n) read as invariant operators (Theorems 9.5.1 and 9.6.1).

A singular vector of I(V) of degree m and weight nu is an invariant operator of order m
whose source is named by the highest weight of V and whose target is named by nu.  The
four lists print the source; naming the operator needs the target too.

This script checks, family by family, the target weights that Remark 9.5.2
asserts for m(n) -- i.e. that in the pe(n)-weights, source and target both, the four
families of Theorem 9.5.1 are the five families of Theorem 9.2.1:

    1a)  (2^{k-1},1^{n-k+1}) -> (2^k,1^{n-k})              = 1a) over le(n)
    1b)  (0^{k-1},L+1,1^{n-k}) -> (0^{k-1},L,1^{n-k})      = 1b) at k=1, 1c) at k=n
    2a)  (1,...,1) -> (1,...,1)                            = 2)  over le(n)
    2b)  (1,1) -> (0,0)   [n=2]                            = 2') over le(n)

For every line: the singular space at the asserted target is computed and required to be
non-zero, and the z-eigenvalue of the target is required to be c-m.

Run:  python3 opsm.py
"""
import sys
from mimod import Mn, MInduced

OK = FAIL = 0


def check(msg, cond):
    global OK, FAIL
    if cond:
        OK += 1
        print("  PASS  " + msg)
    else:
        FAIL += 1
        print("  FAIL  " + msg)


def sing(n, lam, c, m, nu):
    I = MInduced(Mn(n), lam, c)
    return I.singular(m, tuple(nu)), I.zwt(m)


def main():
    for n in (2, 3):
        print("=" * 70)
        print("n = %d" % n)
        # 1a) source (2^{k-1},1^{n-k+1}; n-k+3) -> target (2^k,1^{n-k}); order 1
        for k in range(2, n + 1):
            lam = [2] * (k - 1) + [1] * (n - k + 1)
            tgt = [2] * k + [1] * (n - k)
            c = n - k + 3
            v, zc = sing(n, lam, c, 1, tgt)
            check("1a) k=%d  (%s;%s) -> (%s;%s)  dim=%d"
                  % (k, lam, c, tgt, zc, len(v)), len(v) == 1 and zc == c - 1)
        # 1b) source (0^{k-1},L+1,1^{n-k}; n-k+L+1) -> target (0^{k-1},L,1^{n-k}); order 1
        for k in range(1, n + 1):
            for L in ((1, 2) if k < n else (0, 1, 2)):
                lam = [0] * (k - 1) + [L + 1] + [1] * (n - k)
                tgt = [0] * (k - 1) + [L] + [1] * (n - k)
                c = n - k + L + 1
                v, zc = sing(n, lam, c, 1, tgt)
                check("1b) k=%d L=%d  (%s;%s) -> (%s;%s)  dim=%d"
                      % (k, L, lam, c, tgt, zc, len(v)), len(v) >= 1 and zc == c - 1)
        # 2a) source (1,...,1; L+2) -> target (1,...,1); order 2, every L
        for L in (-3, -1, 0, 2):
            lam = [1] * n
            c = L + 2
            v, zc = sing(n, lam, c, 2, lam)
            check("2a) L=%d  (%s;%s) -> (%s;%s)  dim=%d"
                  % (L, lam, c, lam, zc, len(v)), len(v) >= 1 and zc == c - 2)
        # 2b) n = 2 only: source (1,1;2) -> target (0,0); order 2
        if n == 2:
            v, zc = sing(2, [1, 1], 2, 2, [0, 0])
            check("2b)      ([1, 1];2) -> ([0, 0];%s)  dim=%d" % (zc, len(v)),
                  len(v) >= 1 and zc == 0)
            # and it is absent for n = 3, which is why the family is n = 2 only
    v, zc = sing(3, [1, 1, 1], 2, 2, [0, 0, 0])
    check("2b) absent for n=3  ([1,1,1];2) -> ([0,0,0];%s)  dim=%d" % (zc, len(v)),
          len(v) == 0)

    print("=" * 70)
    print("opsm: %d passed, %d failed" % (OK, FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
