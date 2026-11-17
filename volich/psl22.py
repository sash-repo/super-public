#!/usr/bin/env python3
"""How far does the exception at psl(2|2) go?  (Remark 4.17.1 and the proviso
n >= 3 in the row psl(n|n) of Table 2.12.1.)

Lemma 4.17 establishes that for psl(2|2) with p = q = 1 the element
x^1_1 does NOT ensure nontriviality, although the Table's condition
p + q <= m admits it.  Before the proviso is written into the Table, the row
psl(n|n) has to be looked at whole: it prescribes TWO kinds of homologic
element,

    x^p_q  (the same as for sl(n|n)),   and   x = antidiag(1_n, 1_n),

and the condition column says "as above" for both, although the second carries
no p, q at all.  So the question is whether psl(2|2) drops out of the Table
altogether, or only its x^1_1 does.

Checked here, over Q with Fractions, for n = 2, 3, 4:

  1) x = antidiag(1_n, 1_n) is odd and homologic in psl(n|n)   ([x,x] = 2 x^2
     = 2 * 1_{2n}, which is 0 in the quotient by the center);
  2) whether it ensures nontriviality, i.e., whether S(a, b) = [[a,x],[b,x]]
     is not identically 0 on g_ev;
  3) for n = 2, the same for every x^p_q the Table admits (only p = q = 1),
     so that the whole row is decided at its smallest member.

Run:  python3 psl22.py
"""

import sys

import nontriv                                       # noqa: F401  (models)
from onesided2 import (E, sl_gens, x_sl, subalgebra, identity, _coords_of)
from fractions import Fraction as F


def antidiag_ones(n):
    """x = antidiag(1_n, 1_n) inside gl(n|n)."""
    d = 2 * n
    A = [[F(0)] * d for _ in range(d)]
    for i in range(n):
        A[i][n + i] = F(1)
        A[n + i][i] = F(1)
    return A


def report(nm, n, X):
    d = 2 * n
    alg, _, rows, N = subalgebra(nm, d, n, sl_gens(n, n), center=[identity(d)])
    xv = _coords_of(rows, N, d, X)
    br = alg.br(xv, xv)
    homologic = not any(br)
    h = nontriv.Voli(alg, xv)
    closed = h.closure_ok()
    ens = any(any(h.Sm[i][j]) for i in range(h.dim) for j in range(h.dim))
    print(f"   {nm:26s} homologic: {str(homologic):5s}  closed: "
          f"{str(closed):5s}  dim h = {h.dim:3d}  "
          f"ensures nontriviality: {ens}")
    return ens


def main():
    print("The row psl(n|n) of Table 2.12.1, both kinds of x.\n")

    print(" x = antidiag(1_n, 1_n)  (the element the row lists 'and also'):")
    anti = {}
    for n in (2, 3, 4):
        anti[n] = report(f"psl({n}|{n}) antidiag(1,1)", n, antidiag_ones(n))

    print("\n x = x^p_q, every (p, q) the Table admits (p, q > 0, p+q <= n):")
    xpq = {}
    for n in (2, 3):
        for p in range(1, n):
            for q in range(1, n):
                if p + q <= n:
                    xpq[(n, p, q)] = report(f"psl({n}|{n}) x^{p}_{q}", n,
                                            x_sl(n, n, p, q))

    print()
    if not anti[2] and not xpq[(2, 1, 1)] and anti[3] and xpq[(3, 1, 1)]:
        print(" => BOTH"
              " homologic elements")
        print("    the row prescribes fail at n = 2, so psl(2|2) drops out of"
              " the row")
        print("    altogether, while n = 3 and n = 4 are unaffected in both"
              " columns.")
        print("    (That every OTHER homologic element of psl(2|2) fails as"
              " well is proved")
        print("     by hand in Lemma 4.17: S(a,b) is always a scalar matrix,"
              " by the")
        print("     identity MN + NM = tr(MN)1 for traceless 2x2 matrices.)")
        return 0
    print(" => the reading assumed in the text does not reproduce; do not"
          " write the proviso.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
