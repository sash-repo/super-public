"""
omhom.py -- exhibits the homology of (Omega(const), omega.) explicitly, i.e. the
bottom vertex of the graph of [L3, Subsec. 2.4 b)].

Usage:  python3 omhom.py [nmax]
"""

import sys
from fractions import Fraction as F
from omcheck import basis, omega_mul, to_vec, nullspace, weight
from omirr import rref, reduce_mod


def show(n, m):
    A, B = m
    parts = ["'x_%d" % (a + 1) for a in A]
    for j in range(n):
        if B[j] == 1:
            parts.append("'xi_%d" % (j + 1))
        elif B[j] > 1:
            parts.append("'xi_%d^%d" % (j + 1, B[j]))
    return "".join(parts) if parts else "1"


def run(n, imax):
    bas = {i: basis(n, i) for i in range(0, imax + 3)}
    idx = {i: {m: k for k, m in enumerate(bas[i])} for i in bas}
    print("n = %d" % n)
    for i in range(imax + 1):
        rows = [to_vec(omega_mul(n, {m: F(1)}), idx[i + 2]) for m in bas[i]]
        ker = nullspace([[rows[k][c] for k in range(len(bas[i]))]
                         for c in range(len(bas[i + 2]))], len(bas[i]))
        prev = []
        if i >= 2:
            prows = [to_vec(omega_mul(n, {m: F(1)}), idx[i]) for m in bas[i - 2]]
            prev = rref(prows)[0]
        ech, piv = rref(prev)
        cls = [reduce_mod(v, ech, piv) for v in ker]
        cls = rref([c for c in cls if any(c)])[0]
        if cls:
            for v in cls:
                terms = ["%s*%s" % (c, show(n, m))
                         for m, c in zip(bas[i], v) if c]
                w = {weight(m) for m, c in zip(bas[i], v) if c}
                print("  degree %d: homology class  %s      (weight %s)"
                      % (i, " + ".join(terms), w.pop() if len(w) == 1 else "?"))
    print()


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for n in range(1, nmax + 1):
        run(n, 2 * n + 1)
