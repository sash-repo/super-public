"""What kind of "algebra" does Gz define on the space of metrics?

Grozman: "a 2nd order one which determins an 'algebra' structure on the space of
metrics (symmetric forms) on M".  Gz is commutative (checked elsewhere); here we
test the standard identities.

Usage:  python3 algebra.py [m] [nterms-per-entry]
"""

import sys, os, random
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from symp import Symp, random_sym_field, padd, pscale, pzero  # noqa: E402
from verify import build, SOLUTION, iszero  # noqa: E402


def lin(S, parts):
    n = S.n
    acc = [[pzero() for _ in range(n)] for _ in range(n)]
    for c, T in parts:
        for i in range(n):
            for j in range(i, n):
                acc[i][j] = padd(acc[i][j], pscale(T[i][j], c))
                acc[j][i] = acc[i][j]
    return acc


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nt = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nt = nt or None
    S = Symp(m)
    rng = random.Random(4242 + m)
    G = build(S, SOLUTION)

    # the products are of order 2, so a triple product is of order 4: the test
    # fields must carry nonzero derivatives up to that order.
    A = random_sym_field(S, 6, rng, nterms=nt)
    B = random_sym_field(S, 6, rng, nterms=nt)
    C = random_sym_field(S, 6, rng, nterms=nt)
    print(f"m = {m}")

    print(f"  commutative:  {iszero(S, lin(S, [(1, G(A, B)), (-1, G(B, A))]))}")

    assoc = lin(S, [(1, G(G(A, B), C)), (-1, G(A, G(B, C)))])
    print(f"  associative:  {iszero(S, assoc)}")

    # Jordan identity  (A*B)*(A*A) = A*((B)*(A*A))
    AA = G(A, A)
    jor = lin(S, [(1, G(G(A, B), AA)), (-1, G(A, G(B, AA)))])
    print(f"  Jordan:       {iszero(S, jor)}")

    # is the associator alternating (i.e. is it an alternative algebra)?
    alt = lin(S, [(1, G(G(A, A), B)), (-1, G(A, G(A, B)))])
    print(f"  alternative:  {iszero(S, alt)}")

    # does Gz have a unit-like behaviour on the symplectic form itself?
    n = S.n
    Om2 = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Om2[i][i] = {(0,) * n: Fraction(1)}
    T = G(A, Om2)
    print(f"  Gz(A, delta) = 0 ?  {iszero(S, T)}")


if __name__ == "__main__":
    main()
