"""Verify the candidate universal formula for Gz at several m, and compare it
with Grozman's printed m = 1 formula.

The combination below is the one canon.py found at m = 2.  Every summand is an
index expression that makes sense for every m, so the question "is there ONE
formula for all m?" is a computation.

Usage:  python3 verify.py [m] [nterms-per-entry]
"""

import sys, os, random
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from symp import (Symp, enumerate_terms, eval_term_sym, random_sym_field,  # noqa: E402
                  monomial_polys, padd, pscale, pzero)
from explicit import op_H, op_E1, op_E2, op_P  # noqa: E402
from canon import op_D, op_V1, op_V2, op_M, defect  # noqa: E402

# the m = 2 solution, in the mixed basis canon.py used
SOLUTION = [("P", Fraction(1)), ("V1", Fraction(-1)), ("V2", Fraction(-1)),
            ("M", Fraction(-1, 2)),
            (43, Fraction(1, 2)), (55, Fraction(1)), (61, Fraction(1, 2)),
            (64, Fraction(1)), (68, Fraction(1)), (87, Fraction(-1, 2)),
            (118, Fraction(1, 2))]

NAMED = {"H": op_H, "E1": op_E1, "E2": op_E2, "P": op_P,
         "D": op_D, "V1": op_V1, "V2": op_V2, "M": op_M}


def build(S, spec):
    raw = enumerate_terms(2)

    def op(A, B):
        acc = [[pzero() for _ in range(S.n)] for _ in range(S.n)]
        for key, c in spec:
            if isinstance(key, str):
                T = NAMED[key](S, A, B)
            else:
                T = eval_term_sym(S, raw[key], A, B)
            for i in range(S.n):
                for j in range(i, S.n):
                    acc[i][j] = padd(acc[i][j], pscale(T[i][j], c))
                    acc[j][i] = acc[i][j]
        return acc
    return op


def iszero(S, T):
    return not any(T[i][j] for i in range(S.n) for j in range(i, S.n))


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nt = nt or None
    S = Symp(m)
    rng = random.Random(9001 + m)
    GZ = build(S, SOLUTION)

    A = random_sym_field(S, 4, rng, nterms=nt)
    B = random_sym_field(S, 4, rng, nterms=nt)

    print(f"m = {m}")
    T = GZ(A, B)
    print(f"  is it the zero operator? {iszero(S, T)}")
    for deg in (1, 2, 3, 4):
        bad = [f for f in monomial_polys(S, deg) if not iszero(S, defect(S, GZ, f, A, B))]
        print(f"  invariant under ALL Hamiltonians of degree {deg}: "
              f"{'OK' if not bad else f'FAILS ({len(bad)} of them)'}")

    # symmetry
    T1, T2 = GZ(A, B), GZ(B, A)
    print("  Gz(A,A') = Gz(A',A)?  "
          f"{all(not padd(T1[i][j], pscale(T2[i][j], -1)) for i in range(S.n) for j in range(i, S.n))}")

    # comparison with Grozman's m = 1 combination H + E1 + E2 - P
    G1 = [op_H(S, A, B), op_E1(S, A, B), op_E2(S, A, B), op_P(S, A, B)]
    n = S.n
    GR = [[pzero() for _ in range(n)] for _ in range(n)]
    for c, T0 in zip((1, 1, 1, -1), G1):
        for i in range(n):
            for j in range(i, n):
                GR[i][j] = padd(GR[i][j], pscale(T0[i][j], c))
                GR[j][i] = GR[i][j]
    for lam in (Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(2), Fraction(-1, 2), Fraction(-2)):
        ok = all(not padd(T1[i][j], pscale(GR[i][j], -lam)) for i in range(n) for j in range(i, n))
        if ok:
            print(f"  Gz  ==  {lam} * (H + E1 + E2 - P)   [Grozman's m=1 shape]")
            break
    else:
        eq = all(not padd(GR[i][j], pzero()) for i in range(n) for j in range(i, n))
        print("  Gz is NOT a multiple of (H + E1 + E2 - P) here")


if __name__ == "__main__":
    main()
