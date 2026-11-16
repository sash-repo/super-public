#!/usr/bin/env python3
r"""The restriction k(2n-1) < k(2n+1), and what Grozman's Sect. 14 would run on.

The reduction that proves "no operators of order > 3" in [G4] restricts a singular
vector to a coordinate subspace E and uses that its components are gl(E)-highest and
singular for the smaller algebra.  In the contact case the candidate restriction is
"drop the last pair p_n, q_n": the generating functions independent of (p_n, q_n)
should form a subalgebra isomorphic to k(2n-1).

Checked here, exactly, for n = 2, 3, 4:

  1) the subspace is closed under the contact bracket of k(2n+1);
  2) on it the bracket of k(2n+1) IS the bracket of k(2n-1), coefficient by
     coefficient -- so the inclusion is a homomorphism of Lie algebras and not merely
     of vector spaces;
  3) it respects the grading (deg t = 2, deg p = deg q = 1), so it is an embedding of
     graded algebras and k_0(2n-1) = csp(2n-2) sits inside k_0(2n+1) = csp(2n);
  4) the frame restricts: e_a for a <= 2n-2 involves no p_n, q_n, and R is the same.

What is NOT checked, because it is the open half stated in the Problem: whether the
restriction of the k_1-singularity condition on a tensor field over k(2n+1) is the
corresponding condition over k(2n-1).
"""

import sys

from contactn import Contact, padd, pmul, pscal, pzero


def monomials(n, dmax):
    """All monomials in (t,p,q) of weighted degree <= dmax, deg t = 2, deg p = deg q = 1."""
    nv = 2 * n + 1
    out = []

    def rec(i, e, d):
        if i == nv:
            out.append(tuple(e))
            return
        w = 2 if i == 0 else 1
        k = 0
        while d + k * w <= dmax:
            rec(i + 1, e + [k], d + k * w)
            k += 1

    rec(0, [], 0)
    return out


def drops_last_pair(e, n):
    """Does the monomial e of k(2n+1) involve neither p_n nor q_n?"""
    return e[n] == 0 and e[2 * n] == 0


def embed(e, n):
    """The exponent tuple of k(2n-1) matching a monomial of k(2n+1) free of p_n,q_n."""
    # k(2n+1): 0 = t, 1..n = p, n+1..2n = q;  k(2n-1): 0 = t, 1..n-1 = p, n..2n-2 = q.
    return (e[0],) + e[1:n] + e[n + 1:2 * n]


def check(n, dmax=4, log=print):
    big, small = Contact(n), Contact(n - 1)
    mons = [e for e in monomials(n, dmax) if drops_last_pair(e, n)]
    log(f"n = {n}: {len(mons)} monomials of weighted degree <= {dmax} free of p_{n},q_{n}")

    closed = agree = graded = 0
    for e1 in mons:
        for e2 in mons:
            f, g = {e1: 1}, {e2: 1}
            br = big.kbracket(f, g)
            # 1) closed
            for e in br:
                assert drops_last_pair(e, n), (e1, e2, e)
            closed += 1
            # 2) the same bracket as in k(2n-1)
            fs, gs = {embed(e1, n): 1}, {embed(e2, n): 1}
            brs = small.kbracket(fs, gs)
            lhs = {embed(e, n): c for e, c in br.items() if c}
            rhs = {e: c for e, c in brs.items() if c}
            assert lhs == rhs, (e1, e2, lhs, rhs)
            agree += 1
            # 3) the grading is the one of k(2n-1)
            def deg(e):
                return 2 * e[0] + sum(e[1:])
            for e in br:
                assert deg(e) == deg(e1) + deg(e2) - 2, (e1, e2, e)
            graded += 1

    log(f"  closed under the bracket:            {closed} pairs, all in the subspace")
    log(f"  bracket of k({2*n+1}) = bracket of k({2*n-1}): {agree} pairs, coefficient by coefficient")
    log(f"  weighted degrees add up minus 2:     {graded} pairs")

    # 4) the frame restricts
    nv = 2 * n + 1
    for a in range(1, 2 * n - 1):
        for e in mons:
            h = {e: 1}
            for ee in big.e(a, h):
                assert drops_last_pair(ee, n), (a, e, ee)
    log(f"  the frame e_1..e_{2*n-2} and R preserve the subspace")
    return closed


def check_odd(n, log=print):
    """The same for k(1|n-1) < k(1|n): drop the last odd indeterminate."""
    import super as sup

    mons = []
    for a in range(3):
        for r in range(n + 1):
            for S in __import__("itertools").combinations(range(1, n + 1), r):
                if n in S:
                    continue
                if 2 * a + r <= 4:
                    mons.append((a, S))
    log(f"k(1|{n}): {len(mons)} monomials of weighted degree <= 4 free of xi_{n}")

    pairs = 0
    for m1 in mons:
        for m2 in mons:
            f, g = sup.mono(*m1), sup.mono(*m2)
            br = sup.kbracket(n, f, g)
            for (a, S), c in br.items():
                assert n not in S, (m1, m2, S)
            brs = sup.kbracket(n - 1, f, g)
            assert {k: v for k, v in br.items() if v} == {k: v for k, v in brs.items() if v}, (m1, m2)
            pairs += 1
    log(f"  closed, and the bracket of k(1|{n}) is that of k(1|{n-1}): {pairs} pairs")
    return pairs


def main():
    total = 0
    for n in (2, 3, 4):
        total += check(n, dmax=4)
        print()
    for n in (2, 3, 4, 5):
        total += check_odd(n)
    print(f"\nALL CHECKS PASS ({total} bracket pairs)")


if __name__ == "__main__":
    sys.exit(main() and 0)
