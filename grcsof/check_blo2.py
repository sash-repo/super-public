#!/usr/bin/env python3
"""The decisive control experiment.

[BLO] give explicit supertransvectants for N = 1 (their (3.2)-(3.3)), for
N = 2 (their (3.11)-(3.12)) and for N = 3 (their (4.1)-(4.2)).  We already know
(check_blo.py) that (3.2)-(3.3) come out invariant in our reading of their
conventions, and (check_blo3.py) that (4.1)-(4.2) do not.  Between the two
sits N = 2: it uses SEVERAL eta's, exactly like N = 3, so if (3.11)-(3.12)
are invariant, then our handling of the multi-eta terms is right and the
failure at N = 3 is [BLO]'s, not ours.
"""
from fractions import Fraction as Fr
from blo import BLO, Gamma, check_invariance, dtn
from superN import add, smul, mul, parity


def J_N2_311(alg, tau, lam, k):
    """(3.11), k in N\\{0}:
        sum_{p<=2} sum_{i+j=k-1} (-1)^p(F) G_{i,j,k-1} eta_p(F^i) eta_p(G^j)
      - sum_{i+j=k}   G_{i,j,k-1} F^i G^j
      + sum_{i+j=k-2} G_{i,j,k-1} eta_1 eta_2(F^i) eta_1 eta_2(G^j)
    """
    e = [alg.eta(i) for i in range(2)]
    e12 = lambda u: e[0](e[1](u))
    kk = int(k)

    def T(u, v):
        s = -1 if parity(u) else 1
        out = {}
        for p in range(2):                              # i+j = k-1
            for i in range(kk):
                j = kk - 1 - i
                out = add(out, smul(s * Gamma(tau, lam, i, j, kk - 1),
                                    mul(e[p](dtn(i)(u)), e[p](dtn(j)(v)))))
        for i in range(kk + 1):                         # i+j = k
            j = kk - i
            out = add(out, smul(-Gamma(tau, lam, i, j, kk - 1),
                                mul(dtn(i)(u), dtn(j)(v))))
        for i in range(max(kk - 1, 0)):                 # i+j = k-2
            j = kk - 2 - i
            if j < 0:
                continue
            out = add(out, smul(Gamma(tau, lam, i, j, kk - 1),
                                mul(e12(dtn(i)(u)), e12(dtn(j)(v)))))
        return out
    return T


def J_N2_312(alg, tau, lam, k):
    """(3.12), the odd companion:
        sum_{i+j=k-1} (-1)^p(F) G_{i+1,j+1,k} (i+1)(j+1)
                      ( eta_1(F^i) eta_2(G^j) - eta_2(F^i) eta_1(G^j) )
      + sum_{i+j=k}   G_{i,j,k} ( i(2lam+k-i) eta_1 eta_2(F^{i-1}) G^j
                                - j(2tau+k-j) F^i eta_1 eta_2(G^{j-1}) )
    """
    e = [alg.eta(i) for i in range(2)]
    e12 = lambda u: e[0](e[1](u))
    kk = int(k)

    def T(u, v):
        s = -1 if parity(u) else 1
        out = {}
        for i in range(kk):                             # i+j = k-1
            j = kk - 1 - i
            c = s * Gamma(tau, lam, i + 1, j + 1, kk) * (i + 1) * (j + 1)
            out = add(out,
                      smul(c, mul(e[0](dtn(i)(u)), e[1](dtn(j)(v)))),
                      smul(-c, mul(e[1](dtn(i)(u)), e[0](dtn(j)(v)))))
        for i in range(kk + 1):                         # i+j = k
            j = kk - i
            g = Gamma(tau, lam, i, j, kk)
            if i >= 1:
                out = add(out, smul(g * i * (2 * lam + kk - i),
                                    mul(e12(dtn(i - 1)(u)), dtn(j)(v))))
            if j >= 1:
                out = add(out, smul(-g * j * (2 * tau + kk - j),
                                    mul(dtn(i)(u), e12(dtn(j - 1)(v)))))
        return out
    return T


def run(tau=Fr(3, 7), lam=Fr(-5, 11), tmax=3):
    alg = BLO(2)
    print("[BLO]'s N=2 supertransvectants, generic weights "
          f"(tau={tau}, lam={lam}):")
    for k in (1, 2, 3, 4):
        # Both (3.11) and (3.12) are EVEN operators F_tau (x) F_lam -->
        # F_{tau+lam+k} (every term carries an even number of eta's), and by
        # Prop. 3.1(i) they SPAN that space for generic weights: dim = 2.
        for name, J in (("(3.11)", J_N2_311), ("(3.12)", J_N2_312)):
            T = J(alg, tau, lam, k)
            zero = not any(T({m: 1}, {m2: 1})
                           for m in alg.monos(tmax) for m2 in alg.monos(tmax))
            bad = check_invariance(alg, T, 2 * k, tau, lam, tmax=tmax)
            print(f"  k={k}  {name}: "
                  f"{'IDENTICALLY ZERO, ' if zero else ''}"
                  f"{'invariant' if not bad else str(len(bad)) + ' failures'}")


if __name__ == '__main__':
    run()
