#!/usr/bin/env python3
"""The degenerate weights, explicitly: where does Theorem 6.1.5 of the text hold, and what
happens where it does not?

It is proved for mu1+mu2 not in Zee_{<=0} (Corollary 7.2.6 for the count,
Theorem 7.2.22 for the shape).  For the other weights it is FALSE, and this
script exhibits how, in the two ways it can fail:

 1) the dimension: at (mu1,mu2) = (0,0) and the shift d = 1 (order k = 1/2)
    the space of invariant bilinear DIFFERENTIAL operators is 2-dimensional
    (fullops.solve, the whole space of Lemma 7.2.1, no class fixed in
    advance), and likewise everywhere in the window (158);

 2) the operator: in that window the printed bracket [-,-]_k of eqs. (133),
    (137) DEGENERATES TO ZERO -- its coefficients carry the factors
    (mu_i + a), which vanish exactly there.  So at those weights the printed
    formulas do not even give one operator, let alone all of them.

    ../venv/bin/python degen0.py > degen0.log
"""
from fractions import Fraction as F

import fullops as FO


# ---------------------------------------------------------- the coefficients
def binom(x, n):
    """C(x, n) for a rational x and an integer n >= 0."""
    if n < 0:
        return F(0)
    out = F(1)
    for i in range(n):
        out *= (x - i)
    for i in range(2, n + 1):
        out /= i
    return out


def gamma(a, kfloor, mu1, mu2):
    return (-1) ** (a + 1) * binom(mu1 + kfloor, kfloor - a) * binom(mu2 + kfloor, a)


def gamma_tilde(a, kfloor, mu1, mu2):
    return (-1) ** (a + 1) * binom(mu1 + kfloor - 1, kfloor - a) \
                           * binom(mu2 + kfloor - 1, a)


def bracket_coeffs(k2, mu1, mu2):
    """All the coefficients of the bracket of order k = k2/2, as a dict.
    k2 even -> eq. (133);  k2 odd -> eqs. (136), (137)."""
    out = {}
    if k2 % 2 == 0:                                   # even bracket, k = k2/2
        k = k2 // 2
        for a in range(0, k):
            out[("D f^(%d) D g^(%d)" % (a, k - 1 - a))] = \
                gamma(a, k - 1, mu1, mu2)
        for a in range(0, k + 1):
            out[("f^(%d) g^(%d)" % (a, k - a))] = -gamma_tilde(a, k, mu1, mu2)
    else:                                             # odd bracket, k = m+1/2
        m = (k2 - 1) // 2
        for a in range(0, m + 1):
            g = gamma(a, m, mu1, mu2)
            out[("f^(%d) D g^(%d)" % (a, m - a))] = g * (mu1 + a)
            out[("D f^(%d) g^(%d)" % (a, m - a))] = -g * (mu2 + m - a)
    return out


def window(m1, m2):
    """The exceptional shifts d of Theorem 7.2.11 for mu_i = -m_i."""
    return range(2 * max(m1, m2) + 1, 2 * (m1 + m2) + 1 + 1)


def main():
    print(f"# exactlin backend: {FO.backend()}")

    print("\n=== 1) the dimension of the whole space of invariant operators")
    N = 4
    for (m1, m2, d) in [(F(1, 31), F(5, 31), 1),        # generic reference
                        (F(0), F(0), 1),
                        (F(0), F(-1), 3),
                        (F(-1), F(-1), 3),
                        (F(-1), F(-1), 4),
                        (F(-1), F(-1), 5)]:
        bas, B = FO.solve(N, m1, m2, F(d))
        print(f"  mu = ({m1}, {m2}), shift d = {d} (order k = {F(d, 2)}): "
              f"dim = {len(B)}")
        for v in B:
            print("       ", FO.show(bas, v))

    print("\n=== 2) does the printed bracket [-,-]_k survive in the window?")
    print("#  mu_i = -m_i;  d = 2k is the shift;  'zero' = every coefficient "
          "of the printed formula vanishes")
    allzero = True
    for m1 in range(0, 5):
        for m2 in range(0, 5):
            for d in window(m1, m2):
                c = bracket_coeffs(d, F(-m1), F(-m2))
                z = all(v == 0 for v in c.values())
                allzero = allzero and z
                if (m1, m2) in ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)):
                    print(f"  m = ({m1},{m2}), d = {d}, k = {F(d,2)}: "
                          f"{'ZERO' if z else 'nonzero: ' + str({a: v for a, v in c.items() if v})}")
    print(f"\n#  every bracket in every window (0 <= m_i <= 4) is zero: {allzero}")

    print("\n=== 3) and just outside the window it does not vanish")
    for (m1, m2) in ((0, 0), (1, 1), (2, 2)):
        lo = 2 * max(m1, m2) + 1
        hi = 2 * (m1 + m2) + 1
        for d in (lo - 1, hi + 1):
            if d < 0:
                continue
            c = bracket_coeffs(d, F(-m1), F(-m2))
            z = all(v == 0 for v in c.values())
            print(f"  m = ({m1},{m2}), d = {d} "
                  f"({'below' if d < lo else 'above'} the window): "
                  f"{'ZERO' if z else 'nonzero'}")


if __name__ == "__main__":
    main()


def equivalence():
    """Is 'the printed bracket vanishes' EQUIVALENT to 'd is in the window'?"""
    print("\n=== 4) vanishing of the printed bracket versus the window")
    bad = []
    for m1 in range(0, 6):
        for m2 in range(0, 6):
            lo, hi = 2 * max(m1, m2) + 1, 2 * (m1 + m2) + 1
            for d in range(0, hi + 6):
                c = bracket_coeffs(d, F(-m1), F(-m2))
                z = all(v == 0 for v in c.values())
                inw = lo <= d <= hi
                if z != inw:
                    bad.append((m1, m2, d, z, inw))
    print(f"#  pairs 0 <= m_i <= 5, shifts 0 <= d <= 2(m1+m2)+6:  "
          f"{'the two coincide' if not bad else 'MISMATCHES: ' + str(bad[:20])}")
    print(f"#  mismatches: {len(bad)}")


equivalence()
