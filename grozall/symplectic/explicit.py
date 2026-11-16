"""The explicit shape of Grozman's operator Gz for arbitrary m.

Four natural Sp(2m)-covariant, translation-invariant bilinear operations of
total order 2 on covariant symmetric 2-tensor fields A, A' ("metrics"):

    g(A,A')   = sum omega^{ka} omega^{lb} A_kl A'_ab          (a function)
    H(A,A')_ij = d_i d_j g(A,A')
    dd(A)      = sum omega^{ik} omega^{jl} d_i d_j A_kl        (a function)
    E1_ij      = dd(A) * A'_ij ,      E2_ij = dd(A') * A_ij
    P_ij       = sym_ij  sum_{l,s} omega^{ls} { A_il , A'_js } (Poisson bracket)

For m = 1 these are exactly the four blocks of Grozman's printed formula
(g = ac'-2bb'+ca',  dd A = a_yy - 2 b_xy + c_xx).

The script solves for the invariant combination and verifies it against all
cubic AND quartic Hamiltonians.

Usage:  python3 explicit.py [m]
"""

import sys, os, random
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank, backend  # noqa: E402

from symp import (Symp, random_sym_field, monomial_polys, padd, pscale,  # noqa: E402
                  pmul, pdiff, pzero, _sum_polys)


# ---------------------------------------------------------------- the pieces


def g_fun(S, A, B):
    n, Om = S.n, S.Om
    terms = []
    for k, a, w1 in S.Omnz:
        for l, b, w2 in S.Omnz:
            terms.append(pscale(pmul(A[k][l], B[a][b]), w1 * w2))
    return _sum_polys(terms)


def dd_fun(S, A):
    terms = []
    for i, k, w1 in S.Omnz:
        for j, l, w2 in S.Omnz:
            terms.append(pscale(pdiff(pdiff(A[k][l], i), j), w1 * w2))
    return _sum_polys(terms)


def op_H(S, A, B):
    g = g_fun(S, A, B)
    n = S.n
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = pdiff(pdiff(g, i), j)
            out[i][j] = s
            out[j][i] = s
    return out


def _scalar_times(S, f, A):
    n = S.n
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = pmul(f, A[i][j])
            out[i][j] = s
            out[j][i] = s
    return out


def op_E1(S, A, B):
    return _scalar_times(S, dd_fun(S, A), B)


def op_E2(S, A, B):
    return _scalar_times(S, dd_fun(S, B), A)


def op_P(S, A, B):
    n = S.n
    raw = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            terms = []
            for l, s_, w in S.Omnz:
                terms.append(pscale(S.poisson(A[i][l], B[j][s_]), w))
            raw[i][j] = _sum_polys(terms)
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = pscale(padd(raw[i][j], raw[j][i]), Fraction(1, 2))
            out[i][j] = s
            out[j][i] = s
    return out


NAMED = [("H", op_H), ("E1", op_E1), ("E2", op_E2), ("P", op_P)]


def defect(S, op, f, A, B):
    T = op(S, A, B)
    lhs = S.lie_sym2(f, T)
    r1 = op(S, S.lie_sym2(f, A), B)
    r2 = op(S, A, S.lie_sym2(f, B))
    n = S.n
    out = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = padd(lhs[i][j], pscale(padd(r1[i][j], r2[i][j]), -1))
            out[i][j] = s
            out[j][i] = s
    return out


def rows_from(S, defects, rows):
    keys = set()
    for D in defects:
        for i in range(S.n):
            for j in range(i, S.n):
                for e in D[i][j]:
                    keys.add((i, j, e))
    for (i, j, e) in sorted(keys):
        rows.append([D[i][j].get(e, Fraction(0)) for D in defects])


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    S = Symp(m)
    rng = random.Random(31415 + m)
    print(f"m = {m}, n = {S.n}, backend = {backend()}")

    # --- sanity: the four pieces must be Sp-covariant and translation-invariant
    lin = monomial_polys(S, 1)
    quad = monomial_polys(S, 2)
    A0 = random_sym_field(S, 3, rng)
    B0 = random_sym_field(S, 3, rng)
    bad = []
    for nm, op in NAMED:
        for f in lin + quad:
            D = defect(S, op, f, A0, B0)
            if any(D[i][j] for i in range(S.n) for j in range(i, S.n)):
                bad.append((nm, f))
    print(f"h_(-1)+h_0 invariance of the four pieces: "
          f"{'OK' if not bad else 'FAILS for ' + str(bad[:3])}")

    # --- solve for the invariant combination, using the cubics
    cub = monomial_polys(S, 3)
    rows = []
    for s in range(2):
        A = random_sym_field(S, 3, rng)
        B = random_sym_field(S, 3, rng)
        for f in cub:
            rows_from(S, [defect(S, op, f, A, B) for _, op in NAMED], rows)
    ns = nullspace(rows)
    print(f"equations: {len(rows)};  invariant combinations among "
          f"{[n for n, _ in NAMED]}: {len(ns)}")
    for v in ns:
        print("   ", {n: str(c) for (n, _), c in zip(NAMED, v)})

    if not ns:
        return

    # --- verify the found operator against the quartics too (h_2), and quintics
    v = ns[0]

    def Gz(S, A, B):
        n = S.n
        acc = [[pzero() for _ in range(n)] for _ in range(n)]
        for c, (_, op) in zip(v, NAMED):
            if c == 0:
                continue
            T = op(S, A, B)
            for i in range(n):
                for j in range(i, n):
                    acc[i][j] = padd(acc[i][j], pscale(T[i][j], c))
                    acc[j][i] = acc[i][j]
        return acc

    for deg in (3, 4, 5):
        ok = True
        A = random_sym_field(S, 4, rng)
        B = random_sym_field(S, 4, rng)
        for f in monomial_polys(S, deg):
            D = defect(S, Gz, f, A, B)
            if any(D[i][j] for i in range(S.n) for j in range(i, S.n)):
                ok = False
                break
        print(f"  invariance under ALL Hamiltonians of degree {deg}: "
              f"{'OK' if ok else 'FAILS'}")

    # --- symmetry of the operator
    A = random_sym_field(S, 3, rng)
    B = random_sym_field(S, 3, rng)
    T1 = Gz(S, A, B)
    T2 = Gz(S, B, A)
    sym = all(not padd(T1[i][j], pscale(T2[i][j], -1))
              for i in range(S.n) for j in range(i, S.n))
    ask = all(not padd(T1[i][j], T2[i][j])
              for i in range(S.n) for j in range(i, S.n))
    print(f"  Gz(A,A') = Gz(A',A)?  {sym};   = -Gz(A',A)?  {ask}")


if __name__ == "__main__":
    main()
