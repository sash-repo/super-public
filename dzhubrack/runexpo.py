#!/usr/bin/env python3
"""Theorem 1.6 of arXiv:1906.01380 checked, and Problem 1.6.1 1) tested against
the exponents."""

import sys
import numpy as np

import salg
from expo import survey

EXPONENTS = {
    "sl(2)": [1], "sl(3)": [1, 2], "sl(4)": [1, 2, 3], "sl(5)": [1, 2, 3, 4],
    "o(3)": [1], "o(5)": [1, 3], "o(7)": [1, 3, 5], "o(9)": [1, 3, 5, 7],
    "o(4)": [1, 1], "o(6)": [1, 3, 2], "o(8)": [1, 3, 5, 3], "o(10)": [1, 3, 5, 7, 4],
    "sp(4)": [1, 3], "sp(6)": [1, 3, 5], "sp(8)": [1, 3, 5, 7],
    "g2": [1, 5],
}


def mats(basis):
    return [np.array([[int(x) for x in row] for row in M], dtype=np.int64)
            for M, p in basis]


def sl_basis(n):
    out = []
    for i in range(n):
        for j in range(n):
            M = np.zeros((n, n), dtype=np.int64)
            if i != j:
                M[i][j] = 1
                out.append(M)
    for i in range(n - 1):
        M = np.zeros((n, n), dtype=np.int64)
        M[i][i], M[i + 1][i + 1] = 1, -1
        out.append(M)
    return out


CASES = []
for n in (2, 3, 4, 5):
    CASES.append((f"sl({n})", sl_basis(n), range(2, 2 * n + 2)))
for N in (3, 4, 5, 6, 7, 8):
    b, _ = salg.o(N)
    top = 2 * N + 2
    CASES.append((f"o({N})", mats(b), range(2, top)))
for N in (4, 6, 8):
    b, _ = salg.sp(N)
    CASES.append((f"sp({N})", mats(b), range(2, 2 * N + 2)))
CASES.append(("g2", mats(salg.g2()[0]), range(2, 13)))


if __name__ == "__main__":
    want = sys.argv[1:] or [c[0] for c in CASES]
    for name, basis, ks in CASES:
        if name not in want:
            continue
        print(f"{name}: dim = {len(basis)}, matrix size {basis[0].shape[0]}, "
              f"exponents {EXPONENTS.get(name)}")
        res = survey(name, basis, ks)
        good = [k for k in ks if res[k][0] and res[k][1]]
        ex = EXPONENTS.get(name)
        if ex:
            dbl = sorted(set(2 * m for m in ex))
            print(f"  {name:10s}     doubled exponents {dbl}")
            print(f"  {name:10s}     2m-1 for m>1      {sorted(set(2 * m - 1 for m in ex if m > 1))}")
        print()
        sys.stdout.flush()
