#!/usr/bin/env python3
"""osp(2|2)-invariant binary operators over k(1|2): the GENERIC case, all orders.

Theorem 2.9 of the paper, generic weights, orders up to 12.

classify_k12.py answers the same question with all four weights SYMBOLIC, and
is the authority; but its nullspace over QQ(lam_1,nu_1,lam_2,nu_2) gets
expensive past order 6, whereas the GENERIC dimension only needs the rank at a
random rational weight point.  So here the four weights are specialised to
several random points and the rank is taken over QQ with FLINT: that is exact
at each point, and a rank computed at a random point is the generic rank
unless the point is on the (proper, closed) drop locus -- which is why three
independent points are used and the maximum of the three dimensions is not
taken but their MINIMUM, the generic dimension being the smallest one.

    python osp22gen.py [ORDERMAX]        # default 12

Conventions: invar.py / k12.py (= those of [BLS1]), so an operator of degree
s = -k maps F_{lam_1,nu_1} (x) F_{lam_2,nu_2} --> F_{lam_1+lam_2+k, nu_1+nu_2-w},
where w is the K_{xi eta}-charge.
"""
import sys
from fractions import Fraction

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "lib"))
from exactlin import rank as exact_rank                       # noqa: E402
from sympy import Rational, ilcm, symbols                           # noqa: E402

from invar import gen_binary, eqs_binary, matrix_of           # noqa: E402

l1, n1, l2, n2 = symbols('lambda_1 nu_1 lambda_2 nu_2')
FIELDS = ('txi', 'teta')

POINTS = [
    {l1: Rational(1, 3), n1: Rational(1, 5), l2: Rational(2, 7), n2: Rational(3, 11)},
    {l1: Rational(-4, 9), n1: Rational(5, 13), l2: Rational(-6, 17), n2: Rational(7, 19)},
    {l1: Rational(8, 23), n1: Rational(-9, 29), l2: Rational(10, 31), n2: Rational(-11, 37)},
]


def dim_at(M, cs, pt):
    """dim of the solution space at the weight point pt.

    NB the column count is len(cs), NOT M.cols: when the system is EMPTY --
    which is what happens at order 0, the multiplication being invariant with
    no conditions at all -- matrix_of returns a 0 x 0 Matrix and M.cols is 0."""
    ncols = len(cs)
    rows = []
    for i in range(M.rows):
        vals = [Rational(M[i, j].subs(pt)) for j in range(M.cols)]
        if any(v != 0 for v in vals):
            den = 1
            for v in vals:
                den = ilcm(den, v.q)
            rows.append([int(v * den) for v in vals])
    if not rows:
        return ncols
    assert M.cols == ncols, f"matrix has {M.cols} columns, expected {ncols}"
    return ncols - exact_rank(rows)


def main():
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print("osp(2|2) over k(1|2): generic dim of the invariant binary operators")
    print("order k = -s;  w = K_{xi eta}-charge;  target F_{l1+l2+k, n1+n2-w}")
    print()
    tot = {}
    for s in range(0, -kmax - 1, -1):
        k = -s
        line = []
        for w in (-2, -1, 0, 1, 2):
            terms = gen_binary(s, w)
            if not terms:
                continue
            pB = w % 2
            lt, nt = l1 + l2 - s, n1 + n2 - w
            cs, eqs = eqs_binary(terms, l1, n1, l2, n2, lt, nt, pB,
                                 nmax=k // 2 + 2, fields=FIELDS)
            M = matrix_of(eqs, cs)
            dims = [dim_at(M, cs, pt) for pt in POINTS]
            d = min(dims)
            line.append((w, len(terms), d, dims))
        tot[k] = sum(d for _, _, d, _ in line)
        txt = "  ".join(f"w={w:+d}: {nt_:2d} words, dim {d}"
                        + ("" if len(set(dd)) == 1 else f" (points differ: {dd})")
                        for w, nt_, d, dd in line)
        print(f"order {k:2d} (parity {'even' if k % 2 == 0 else 'odd'}): {txt}"
              f"   ||  total {tot[k]}")
        sys.stdout.flush()
    print()
    print("totals per order:", ", ".join(f"k={k}: {v}" for k, v in sorted(tot.items())))


if __name__ == "__main__":
    main()
