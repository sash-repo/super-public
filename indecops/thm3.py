"""Malyshev's Theorem 3, and why it is a two-line statement.

His (15)-(16) are one condition for each p >= 1; his Theorem 3 says that the
conditions for p >= 3 follow from p = 1 and p = 2, and he writes "it is not
excluded that Theorem 3 has a quite simple explanation".  It has: the
condition indexed by p is invariance under the vector field x^{p+1} d/dx,
and

    [x^2 d/dx, x^m d/dx] = (m-2) x^{m+1} d/dx,

so x^2 d/dx and x^3 d/dx generate the whole of <x^m d/dx | m >= 2>, and
invariance under a set of fields is invariance under the Lie algebra they
generate.  The same argument covers (16), i.e. arbitrary weights.

Checked here: imposing q = 2, 3 only gives the same space as imposing all q.
"""

import sys
from fractions import Fraction as F
from line import midx, _inv_matrix
from exactlin import rank, nullspace

ok, bad = 0, []
WEIGHTS = [(0,), (1,), (0, 0), (1, 1), (1, 2), (F(-2, 3), F(-2, 3)),
           (0, 0, 0), (1, 1, 1), (2, 1, 1), (1, 2, 3), (0, 0, 0, 0),
           (1, 1, 1, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)]


def restricted(weights, k, qmax):
    """The system with only the conditions q = 2..qmax."""
    n = len(weights)
    cols = midx(n, k)
    pos = {z: j for j, z in enumerate(cols)}
    rows = []
    from line import binom
    for q in range(2, qmax + 1):
        for w in midx(n, k - q + 1):
            row = [F(0)] * len(cols)
            nz = False
            for i in range(n):
                z = list(w)
                z[i] += q - 1
                c = F(binom(w[i] + q - 1, q)) + weights[i] * F(
                    binom(w[i] + q - 1, q - 1))
                if c:
                    row[pos[tuple(z)]] += c
                    nz = True
            if nz:
                rows.append(row)
    if not rows:
        return len(cols)
    return len(cols) - rank(rows)


for w in WEIGHTS:
    w = tuple(F(a) for a in w)
    n = len(w)
    for k in range(0, min(12, 3 * n + 4)):
        cols, rows = _inv_matrix(w, k)
        full = len(cols) - (rank(rows) if rows else 0)
        two3 = restricted(w, k, 3)
        if full == two3:
            ok += 1
        else:
            bad.append(f"weights {w}, k={k}: all q gives {full}, q<=3 gives {two3}")
    # and q = 2 alone is NOT enough (so the pair really is needed)
for w in [(F(1), F(1), F(1)), (F(0), F(0), F(0))]:
    diff = [k for k in range(2, 8)
            if restricted(w, k, 2) != restricted(w, k, 3)]
    if diff:
        ok += 1
    else:
        bad.append(f"q=2 alone happened to suffice for {w}")

print(f"{ok} checks passed, {len(bad)} failed")
for b in bad:
    print("  FAIL", b)
