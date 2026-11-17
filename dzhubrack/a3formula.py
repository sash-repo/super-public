#!/usr/bin/env python3
r"""Formula (19) of the text: on the module F_lambda of weighted densities,

    a_3(rho(X_1), rho(X_2), rho(X_3)) = -lambda(lambda-1) W(u_1,u_2,u_3) * Id,
    X_i = u_i d/dt,   W = det(u_i, u_i', u_i'')  (the Wronskian).

Checked exactly over Q[lambda] on the basis u_i = t^{m_i}, 0 <= m_1 < m_2 < m_3 <= M
(a_3 is trilinear and alternating, so this is the whole claim on polynomial
vector fields): both sides are polynomials in lambda of degree <= 2 with
operator coefficients, so it suffices to compare them at the three values
lambda = 0, 1, 2 -- we compare at five.
"""

import sys
from fractions import Fraction
from itertools import combinations

from sant import amult
from densities import dmul, dadd, rho


def wronskian(ms):
    # u = t^m:  u = t^m, u' = m t^{m-1}, u'' = m(m-1) t^{m-2}; W = c t^{sum m - 3}
    a, b, c = ms
    rows = [[1, 1, 1], [a, b, c], [a * (a - 1), b * (b - 1), c * (c - 1)]]
    det = (rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
           - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
           + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]))
    return det, a + b + c - 3


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    lams = [Fraction(0), Fraction(1), Fraction(2), Fraction(-1), Fraction(3, 7)]
    bad = cnt = 0
    for ms in combinations(range(M + 1), 3):
        det, e = wronskian(ms)
        for lam in lams:
            A = amult([rho(m, lam) for m in ms], [0, 0, 0], dmul, dadd, {})
            coeff = -lam * (lam - 1) * det
            want = {(e, 0): coeff} if coeff != 0 else {}
            cnt += 1
            if A != want:
                bad += 1
                print("MISMATCH", ms, lam, A, want)
    print(f"a_3 = -lambda(lambda-1) W * Id: {cnt} cases (m_1<m_2<m_3<={M}, "
          f"5 values of lambda), {bad} mismatches")
