#!/usr/bin/env python3
"""dim of the centroid {C in End L : C ad(x) = ad(x) C} over F_2; for simple L,
L is absolutely simple iff this is 1."""
import sys
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace
from tab2txt import load
for p in sys.argv[1:]:
    tab = load(p); n = len(tab)
    # unknown C[r][c] (row r, col c), C e_c = sum_r C[r][c] e_r ; index r*n+c
    rows = []
    for i in range(n):
        A = [[(tab[i][c] >> r) & 1 for c in range(n)] for r in range(n)]  # ad e_i
        for r in range(n):
            for c in range(n):
                row = [0] * (n * n)
                for k in range(n):      # (C A)[r][c] - (A C)[r][c]
                    if A[k][c]: row[r * n + k] ^= 1
                    if A[r][k]: row[k * n + c] ^= 1
                if any(row): rows.append(row)
    print(p, 'dim L', n, 'dim centroid', len(nullspace(rows, 2)))
