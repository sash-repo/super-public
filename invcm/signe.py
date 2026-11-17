#!/usr/bin/env python3
"""Corollary 3.2.2 (and the hypotheses of Theorem 3.2.1) for the 370 Cartan matrices of
Subsections 9.1 and 9.2.

For each matrix A, normalized as printed (A_ij <= 0 for i != j), we check:

  * A is a Z-matrix (all elements outside the main diagonal are <= 0);
  * every proper main minor of A is >= 0, so that every main submatrix obtained by deleting
    one node is a direct sum of matrices of finite or affine type;
  * A is of indefinite type in Vinberg's sense: there is u > 0 with Au < 0.  An exact witness
    is u = -A^{-1}(1,...,1)^T, for which Au = -(1,...,1)^T; we check u > 0 over Q;
  * det A < 0;
  * as a floating-point cross-check, the smallest real eigenvalue lambda_0(A) of
    formula (29) is negative.

Usage:  python signe.py
"""
import itertools
from fractions import Fraction

import numpy as np

from symm import symmetrizer
from tables import load, det, mul

stats = {}
bad = []
lam_max = None
for e in load():
    A, B, name = e["A"], e["inverse"], e["name"]
    n = len(A)
    Z = all(A[i][j] <= 0 for i in range(n) for j in range(n) if i != j)
    minors_ok = all(det([[A[i][j] for j in S] for i in S]) >= 0
                    for k in range(1, n) for S in itertools.combinations(range(n), k))
    u = [-sum(B[i][j] for j in range(n)) for i in range(n)]          # u = -A^{-1} 1
    Au = [r[0] for r in mul(A, [[x] for x in u])]
    witness = all(x > 0 for x in u) and all(x == -1 for x in Au)
    ev = np.linalg.eigvals(np.array([[float(x) for x in r] for r in A]))
    lam0 = min(z.real for z in ev if abs(z.imag) < 1e-9)
    lam_max = lam0 if lam_max is None else max(lam_max, lam0)
    key = (e["subsection"], symmetrizer(A) is not None)
    stats[key] = stats.get(key, 0) + 1
    if not (Z and minors_ok and witness and det(A) < 0 and lam0 < 0):
        bad.append((name, Z, minors_ok, witness, det(A), lam0))

print("matrices checked:", sum(stats.values()))
for (sect, sym), v in sorted(stats.items()):
    print("  Subsection %s, %s: %d" % (sect, "symmetrizable" if sym else "non-symmetrizable", v))
print("largest lambda_0 met: %.4f" % lam_max)
print("violations of (Z-matrix, proper main minors >= 0, u > 0 with Au < 0, det < 0, "
      "lambda_0 < 0): %d" % len(bad))
for b in bad[:20]:
    print("  ", b)
