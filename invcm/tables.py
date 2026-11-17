"""The 370 Cartan matrices of Subsections 9.1 and 9.2, read from tables.json.

tables.json is a transcription of the tables of Subsection 9.1 (the 132 almost affine
Lie superalgebras) and Subsection 9.2 (the 238 hyperbolic Lie algebras) of the text.
Each entry holds, as printed:

    subsection   "9.1" or "9.2"
    name         the name of the algebra (NS3_2, S3_35, H4_12, ...)
    marked       True if the entry carries the mark !!! (inverse with no zero element)
    A            the Cartan matrix, entries as rational numbers in string form
    factor       the scalar printed in front of the inverse matrix
    inverse      the matrix printed after that scalar
    det          the determinant printed in the last column

Exact rational arithmetic (fractions.Fraction) throughout.
"""
import json
import os
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path=None):
    """Return a list of dicts; A, inverse (with the factor applied) and det as Fractions."""
    with open(path or os.path.join(HERE, "tables.json"), encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for e in raw:
        fac = Fraction(e["factor"])
        out.append({
            "subsection": e["subsection"],
            "name": e["name"],
            "marked": e["marked"],
            "A": [[Fraction(x) for x in r] for r in e["A"]],
            "inverse": [[Fraction(x) * fac for x in r] for r in e["inverse"]],
            "det": Fraction(e["det"]),
        })
    return out


def det(M):
    """Determinant by Gaussian elimination over Q."""
    n = len(M)
    M = [row[:] for row in M]
    d = Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
            d = -d
        d *= M[c][c]
        inv = Fraction(1) / M[c][c]
        for r in range(c + 1, n):
            if M[r][c]:
                f = M[r][c] * inv
                for k in range(c, n):
                    M[r][k] -= f * M[c][k]
    return d


def mul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
