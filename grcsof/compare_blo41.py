#!/usr/bin/env python3
"""[BLO]'s (4.1) (the ODD supertransvectants, 2k odd) against the truth.

Every term of (4.1) has the shape of one of our ansatz terms
    (a, A, b, B)  <-->  d_x^a eta_A (x) d_x^b eta_B,
and [BLO]'s (-1)^{p(F)} appears exactly on those groups whose second factor
carries an odd number of eta's -- i.e. their sign convention IS the Koszul
convention built into our ansatz.  So the printed coefficients of (4.1) can be
read off into the ansatz basis and compared, term by term, with the unique
invariant operator that the classifier finds.
"""
import sys
from fractions import Fraction as Fr
from blo import BLO, Classifier, Gamma, Lam, terms_of_order, label

PAIRS = [(0, 1), (0, 2), (1, 2)]                   # 1<=p<l<=3, 0-based


def coeffs_41(tau, lam, k, ts):
    """[BLO]'s (4.1) as a coefficient vector in the ansatz basis `ts`."""
    idx = {t: i for i, t in enumerate(ts)}
    c = [0] * len(ts)
    kb = int(k // 1)                                # [k]
    for i in range(kb):                             # i + j = [k] - 1
        j = kb - 1 - i
        if j < 0:
            continue
        for (p, l) in PAIRS:
            q = 3 - p - l
            # - Gamma_{i+1,j+2,k+1} Lam^i_j  eta_q(F^(i)) eta_p eta_l(G^(j))
            c[idx[(i, (q,), j, (p, l))]] += \
                -Gamma(tau, lam, i + 1, j + 2, k + 1) * Lam(i, j)
            # - Gamma_{i+2,j+1,k+1} Lam^j_i  eta_p eta_l(F^(i)) eta_q(G^(j))
            c[idx[(i, (p, l), j, (q,))]] += \
                -Gamma(tau, lam, i + 2, j + 1, k + 1) * Lam(j, i)
        # + Gamma_{i,j+3,k+1} Lam^j_{j+1}  F^(i) eta_1 eta_2 eta_3(G^(j))
        c[idx[(i, (), j, (0, 1, 2))]] += \
            Gamma(tau, lam, i, j + 3, k + 1) * Lam(j, j + 1)
        # + Gamma_{i+3,j,k+1} Lam^i_{i+1}  eta_1 eta_2 eta_3(F^(i)) G^(j)
        c[idx[(i, (0, 1, 2), j, ())]] += \
            Gamma(tau, lam, i + 3, j, k + 1) * Lam(i, i + 1)
    return c


def run(k, tau=Fr(3, 7), lam=Fr(-5, 11)):
    alg = BLO(3)
    ordr = int(2 * k)
    ts, ns = Classifier(alg, tau, lam).solve(ordr)
    print(f"\nk = {k} (order {ordr}), tau={tau}, lam={lam}: "
          f"true space is {len(ns)}-dimensional")
    if len(ns) != 1:
        return
    true = ns[0]
    theirs = coeffs_41(tau, lam, k, ts)
    # normalize both on the first term where the true operator is non-zero
    piv = next(i for i, x in enumerate(true) if x != 0)
    scale = (Fr(theirs[piv]) / true[piv]) if true[piv] and theirs[piv] else None
    print(f"{'term':>16} | {'true (scaled)':>22} | {'[BLO] (4.1)':>22} | ok")
    print("-" * 76)
    nbad = 0
    for i, t in enumerate(ts):
        a, b = true[i], theirs[i]
        if a == 0 and b == 0:
            continue
        sa = a * scale if scale is not None else a
        good = (sa == b)
        nbad += 0 if good else 1
        print(f"{label(t):>16} | {str(sa):>22} | {str(b):>22} | "
              f"{'' if good else 'DIFFERS'}")
    print(f"  -> {'identical up to scale' if not nbad else str(nbad) + ' terms differ'}")


if __name__ == '__main__':
    for k in [Fr(x) for x in (sys.argv[1:] or ['3/2', '5/2'])]:
        run(k)
