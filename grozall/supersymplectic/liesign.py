"""The sign in the Lie derivative on the u-encoded tensor fields, decided.

The Lie derivative of a metric (formula for L_X in ssuper.py) carries a sign
factor (-1)^eps(p(i),p(k),p(X)) in front of its second sum, eps being any
Z/2-polynomial in the three parities: 2^8 = 256 candidates.  This script keeps
those for which

  a) L is a representation of h(2m|n) on metrics:
     L_f L_g T - (-1)^{p(f)p(g)} L_g L_f T = L_{{f,g}} T;
  b) L_X(df) = d(X f) on 1-forms df = sum_i u^i d_i f.

Usage:  python3 liesign.py
"""

import random
from itertools import product

from ssuper import SSetup


def survivors(m, n, nsamp=8, seed=5):
    S = SSetup(m, n, "S")
    A = S.A
    rng = random.Random(seed)
    fs = [p for d in (1, 2, 3) for p in S.monomials(d)]
    gens = rng.sample(fs, min(nsamp, len(fs)))
    # the cubic odd monomials make the term p(i)p(k)p(X) of eps visible: they
    # need n >= 3 (a derivative d_i X^k with i, k odd and X odd)
    gens += [p for p in S.monomials(3) if A.parity(p) and p not in gens][:2]
    T = S.random_field(2, rng, nterms=3)
    keep = []
    for bits in product((0, 1), repeat=8):
        S.set_lie_sign(bits)
        ok = True
        for f in gens:
            for g in gens:
                pf, pg = A.parity(f), A.parity(g)
                # b) on 1-forms
                if S.lie(f, S.dform(g)) != S.dform(S.act(f, g)):
                    ok = False
                    break
                # a) representation on metrics
                lhs = A.add(S.lie(f, S.lie(g, T)),
                            A.scale(S.lie(g, S.lie(f, T)),
                                    -1 if not (pf * pg) % 2 else 1))
                if lhs != S.lie(S.bracket(f, g), T):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            keep.append(bits)
    S.set_lie_sign((0,) * 8)
    return keep


def main():
    common = None
    for m, n in [(1, 1), (1, 2), (2, 1), (1, 3)]:
        k = survivors(m, n)
        print(f"  (m,n)=({m},{n}): {len(k)} of 256 sign patterns survive: {k}")
        common = set(k) if common is None else common & set(k)
    print("common to all (m,n):", sorted(common))
    print("PASS: exactly eps = 0" if sorted(common) == [(0,) * 8] else "FAIL")


if __name__ == "__main__":
    main()
