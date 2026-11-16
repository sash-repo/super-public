#!/usr/bin/env python3
"""The corrected forms of [BLO]'s (4.1) and (4.2), and their verification.

(4.2), even k:  the four groups (r = 0,1,2,3 eta's on each argument) must carry
    the signs  eps_r = -(-1)^{r(r+1)/2} = (-, +, +, -),
which is exactly the pattern of [BLO]'s own (3.3) (N=1) and (3.11) (N=2);
as printed, (4.2) has (-, -, -, +).

(4.1), semi-integer k:  the two sums over 1<=p<l<=3 must carry the Levi-Civita
    sign eps_{p,l,q},  q = 6-l-p,
which is +1 for (p,l) = (1,2), (2,3) and -1 for (1,3); as printed, (4.1) gives
the same coefficient to all three.  (The last two sums then need an overall
minus, which is only a normalization.)

Both corrected operators are checked here against ALL generators of osp(3|2).
"""
import sys
from fractions import Fraction as Fr
from blo import BLO, Gamma, Lam, check_invariance, dtn
from superN import add, smul, mul, parity

PAIRS = [(0, 1), (0, 2), (1, 2)]                    # 1<=p<l<=3, 0-based
EPS = {(0, 1): 1, (0, 2): -1, (1, 2): 1}            # eps_{p,l,6-l-p}
SIGNS = (-1, 1, 1, -1)                             # eps_r for (4.2)


def J42_fixed(alg, tau, lam, k):
    e = [alg.eta(i) for i in range(3)]
    e123 = lambda u: e[0](e[1](e[2](u)))
    kk = int(k)

    def T(u, v):
        s = -1 if parity(u) else 1                  # (-1)^{p(F)}
        out = {}
        for i in range(kk + 1):                     # r = 0, i+j = k
            j = kk - i
            out = add(out, smul(SIGNS[0] * Gamma(tau, lam, i, j, kk - 1),
                                mul(dtn(i)(u), dtn(j)(v))))
        for p in range(3):                          # r = 1, i+j = k-1
            for i in range(kk):
                j = kk - 1 - i
                out = add(out, smul(SIGNS[1] * s * Gamma(tau, lam, i, j, kk - 1),
                                    mul(e[p](dtn(i)(u)), e[p](dtn(j)(v)))))
        for (p, l) in PAIRS:                        # r = 2, i+j = k-2
            for i in range(max(kk - 1, 0)):
                j = kk - 2 - i
                if j < 0:
                    continue
                w = lambda z: e[p](e[l](z))
                out = add(out, smul(SIGNS[2] * Gamma(tau, lam, i, j, kk - 1),
                                    mul(w(dtn(i)(u)), w(dtn(j)(v)))))
        for i in range(max(kk - 2, 0)):             # r = 3, i+j = k-3
            j = kk - 3 - i
            if j < 0:
                continue
            out = add(out, smul(SIGNS[3] * s * Gamma(tau, lam, i, j, kk - 1),
                                mul(e123(dtn(i)(u)), e123(dtn(j)(v)))))
        return out
    return T


def J41_fixed(alg, tau, lam, k):
    e = [alg.eta(i) for i in range(3)]
    e123 = lambda u: e[0](e[1](e[2](u)))
    kb = int(k // 1)                                # [k]

    def T(u, v):
        s = -1 if parity(u) else 1
        out = {}
        for i in range(kb):                         # i + j = [k] - 1
            j = kb - 1 - i
            if j < 0:
                continue
            for (p, l) in PAIRS:
                q = 3 - p - l
                ep = EPS[(p, l)]
                out = add(out, smul(
                    ep * Gamma(tau, lam, i + 1, j + 2, kb + 1) * Lam(i, j),
                    mul(e[q](dtn(i)(u)), e[p](e[l](dtn(j)(v))))))
                out = add(out, smul(
                    ep * s * Gamma(tau, lam, i + 2, j + 1, kb + 1) * Lam(j, i),
                    mul(e[p](e[l](dtn(i)(u))), e[q](dtn(j)(v)))))
            out = add(out, smul(
                s * Gamma(tau, lam, i, j + 3, kb + 1) * Lam(j, j + 1),
                mul(dtn(i)(u), e123(dtn(j)(v)))))
            out = add(out, smul(
                Gamma(tau, lam, i + 3, j, kb + 1) * Lam(i, i + 1),
                mul(e123(dtn(i)(u)), dtn(j)(v))))
        return out
    return T


def run(tau=Fr(3, 7), lam=Fr(-5, 11), tmax=3):
    alg = BLO(3)
    print(f"corrected [BLO] osp(3|2) operators, tau={tau}, lam={lam}")
    for k, J, name in ([(k, J42_fixed, "(4.2)") for k in (1, 2, 3, 4)] +
                       [(Fr(2 * x + 3, 2), J41_fixed, "(4.1)") for x in range(3)]):
        T = J(alg, tau, lam, k)
        ordr = int(2 * k)
        zero = not any(T({m: 1}, {m2: 1})
                       for m in alg.monos(tmax) for m2 in alg.monos(tmax))
        bad = check_invariance(alg, T, ordr, tau, lam, tmax=tmax)
        print(f"  {name} k={k} (order {ordr}): "
              f"{'IDENTICALLY ZERO, ' if zero else 'nonzero, '}"
              f"{'invariant' if not bad else str(len(bad)) + ' FAILURES'}")
        sys.stdout.flush()


if __name__ == '__main__':
    run()
