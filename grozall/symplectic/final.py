"""The formula as it is printed in the text, coded independently from the
index expressions written there, and checked against the engine's answer.

    <A,A'>      = sum omega^{ka} omega^{lb} A_kl A'_ab              (function)
    (Dv A)_i    = sum omega^{pl} d_p A_{il}                         (1-form)
    dsq A       = sum omega^{ik} d_i (Dv A)_k                       (function)
    zeta_A^p    = sum omega^{pr} (Dv A)_r                           (vector field)
    Delta_A     = sum omega^{kp} omega^{lq} A_kl d_p d_q            (2nd order operator)

    P_ij  = sym  sum omega^{ls} { A_il , A'_js }
    G1_ij = Delta_A(A'_ij) + Delta_{A'}(A_ij)
    G2_ij = zeta_{A'}(A_ij) + zeta_A(A'_ij)
    G3_ij = sym sum omega^{kq}omega^{ls} d_i A_kl d_q A'_js
          + sym sum omega^{pr}omega^{ls} d_p A_il d_j A'_rs
    G4_ij = sym sum omega^{ps}omega^{lq} d_p A_il d_q A'_js
    G5_ij = sym sum omega^{ls} ( d_i(Dv A)_l A'_js + d_i(Dv A')_l A_js )
    M_ij  = sym sum omega^{kr}omega^{ls} d_i A_kl d_j A'_rs

    Gz = 2P + G1 + G2 + 2G3 + 2G4 - 2G5 - M

Usage:  python3 final.py [m] [nterms-per-entry]
"""

import sys, os, random
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from symp import (Symp, random_sym_field, monomial_polys, padd, pscale,  # noqa: E402
                  pmul, pdiff, pzero, _sum_polys)
from verify import build, SOLUTION, iszero  # noqa: E402
from canon import defect  # noqa: E402


def Dv(S, A):
    """(Dv A)_i = sum omega^{pl} d_p A_{il}."""
    return [_sum_polys([pscale(pdiff(A[i][l], p), w) for p, l, w in S.Omnz])
            for i in range(S.n)]


def zeta(S, A):
    d = Dv(S, A)
    return [_sum_polys([pscale(d[r], w) for p2, r, w in S.Omnz if p2 == p])
            for p in range(S.n)]


def sym(S, raw):
    n = S.n
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = pscale(padd(raw[i][j], raw[j][i]), Fraction(1, 2))
            out[i][j] = s
            out[j][i] = s
    return out


def piece_P(S, A, B):
    n = S.n
    raw = [[_sum_polys([pscale(S.poisson(A[i][l], B[j][s_]), w)
                        for l, s_, w in S.Omnz]) for j in range(n)] for i in range(n)]
    return sym(S, raw)


def Delta(S, A, f):
    """Delta_A applied to the single function f."""
    return _sum_polys([pscale(pmul(A[k][l], pdiff(pdiff(f, p), q)), w1 * w2)
                       for k, p, w1 in S.Omnz for l, q, w2 in S.Omnz])


def piece_G1(S, A, B):
    n = S.n
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = padd(Delta(S, A, B[i][j]), Delta(S, B, A[i][j]))
            out[i][j] = s
            out[j][i] = s
    return out


def piece_G2(S, A, B):
    n = S.n
    zA, zB = zeta(S, A), zeta(S, B)
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = _sum_polys([pmul(zB[p], pdiff(A[i][j], p)) for p in range(n)]
                           + [pmul(zA[p], pdiff(B[i][j], p)) for p in range(n)])
            out[i][j] = s
            out[j][i] = s
    return out


def piece_G3(S, A, B):
    n = S.n
    raw = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            t = []
            for k, q, w1 in S.Omnz:
                for l, s_, w2 in S.Omnz:
                    t.append(pscale(pmul(pdiff(A[k][l], i), pdiff(B[j][s_], q)), w1 * w2))
            for p, r, w1 in S.Omnz:
                for l, s_, w2 in S.Omnz:
                    t.append(pscale(pmul(pdiff(A[i][l], p), pdiff(B[r][s_], j)), w1 * w2))
            raw[i][j] = _sum_polys(t)
    return sym(S, raw)


def piece_G4(S, A, B):
    n = S.n
    raw = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            t = []
            for p, s_, w1 in S.Omnz:
                for l, q, w2 in S.Omnz:
                    t.append(pscale(pmul(pdiff(A[i][l], p), pdiff(B[j][s_], q)), w1 * w2))
            raw[i][j] = _sum_polys(t)
    return sym(S, raw)


def piece_G5(S, A, B):
    n = S.n
    dA, dB = Dv(S, A), Dv(S, B)
    raw = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            t = []
            for l, s_, w in S.Omnz:
                t.append(pscale(pmul(pdiff(dA[l], i), B[j][s_]), w))
                t.append(pscale(pmul(pdiff(dB[l], i), A[j][s_]), w))
            raw[i][j] = _sum_polys(t)
    return sym(S, raw)


def piece_M(S, A, B):
    n = S.n
    raw = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            t = []
            for k, r, w1 in S.Omnz:
                for l, s_, w2 in S.Omnz:
                    t.append(pscale(pmul(pdiff(A[k][l], i), pdiff(B[r][s_], j)), w1 * w2))
            raw[i][j] = _sum_polys(t)
    return sym(S, raw)


PIECES = [(2, piece_P), (1, piece_G1), (1, piece_G2), (2, piece_G3),
          (2, piece_G4), (-2, piece_G5), (-1, piece_M)]


def Gz(S, A, B):
    n = S.n
    acc = [[pzero() for _ in range(n)] for _ in range(n)]
    for c, f in PIECES:
        T = f(S, A, B)
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
    rng = random.Random(555 + m)
    A = random_sym_field(S, 4, rng, nterms=nt)
    B = random_sym_field(S, 4, rng, nterms=nt)

    print(f"m = {m}")
    mine = Gz(S, A, B)
    engine = build(S, SOLUTION)(A, B)
    engine = [[pscale(engine[i][j], 2) for j in range(S.n)] for i in range(S.n)]
    same = all(not padd(mine[i][j], pscale(engine[i][j], -1))
               for i in range(S.n) for j in range(i, S.n))
    print(f"  the printed formula equals the engine's answer (x2): {same}")
    print(f"  it is not the zero operator: {not iszero(S, mine)}")
    for deg in (1, 2, 3, 4):
        bad = [f for f in monomial_polys(S, deg)
               if not iszero(S, defect(S, lambda P, Q: Gz(S, P, Q), f, A, B))]
        print(f"  invariant under ALL Hamiltonians of degree {deg}: "
              f"{'OK' if not bad else f'FAILS ({len(bad)})'}")


if __name__ == "__main__":
    main()
