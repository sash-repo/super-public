"""The order-0 invariant operators over an ODD symplectic form, named.

Order 0 means: no derivatives at all, so a scheme is one omega-contraction of the
four fiber slots of the pair (A~,B~).  There are exactly three ways to pair them,

    (Au,Au)   ->  omega^{ij}A_{ij} * B~        the periplectic trace of A, times B
    (Au,Bu)   ->  omega^{ij}A_{ik}B_{jl}u^ku^l  the "product" A o B
    (Bu,Bu)   ->  omega^{ij}B_{ij} * A~        the trace of B, times A

and srun.py finds the space of INVARIANT ones to be 3-dimensional, i.e. all three
are invariant.  (Over an EVEN form the trace vanishes identically -- omega is
antisymmetric there and A super-symmetric -- which is why the same count gives 1.)

Every operator of order 0 is ODD here: one omega-contraction, and omega^{ij} is
non-zero only for p(i)+p(j)=1.

This script names them: it checks that scheme no. s is what the hand-written
formula says, and that the trace ptr(A) = omega^{ij}A_{ij} is not identically 0.
"""

import os
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from ssuper import SSetup                                  # noqa: E402
from sgen import enumerate_schemes, evaluate               # noqa: E402


def ptr(S, Atil):
    """omega^{ij} d_{u^i} d_{u^j} A~ / 2 -- the periplectic trace, by hand."""
    A = S.A
    out = []
    for (i, j, w) in S.Omnz:
        out.append(A.scale(S.du(S.du(Atil, i), j), w))
    return A.scale(A.summ(out), Fraction(1, 2))


def prod(S, Atil, Btil):
    """omega^{ij} (d_{u^i}A~)(d_{u^j}B~) -- the product A o B, by hand."""
    A = S.A
    out = []
    for (i, j, w) in S.Omnz:
        P = S.du(Atil, i)
        if not P:
            continue
        Q = S.du(Btil, j)
        if not Q:
            continue
        out.append(A.scale(A.mul(P, Q), w))
    return A.summ(out)


def main():
    schemes = enumerate_schemes(0)
    print(f"order 0: {len(schemes)} schemes")
    for idx, s in enumerate(schemes):
        print(f"   {idx}: {s}")
    for k in (2, 3):
        S = SSetup(k, k, form="odd")
        A = S.A
        rng = random.Random(1234 + k)
        print(f"\n--- k = {k}, sdim ({k}|{k}) ---")
        for t in range(3):
            P = S.random_field(2, rng, nterms=3)
            Q = S.random_field(2, rng, nterms=3)
            vals = [evaluate(S, s, P, Q) for s in schemes]
            # identify each scheme with the hand-written formula
            tA = ptr(S, P)
            tB = ptr(S, Q)
            byhand = {
                "ptr(A).B": A.mul(tA, Q),
                "A o B": prod(S, P, Q),
                "ptr(B).A": A.mul(tB, P),
            }
            if t == 0:
                for name, v in byhand.items():
                    match = [i for i, w in enumerate(vals)
                             if w == v or w == A.scale(v, -1)
                             or w == A.scale(v, 2) or w == A.scale(v, -2)]
                    print(f"   {name:12s} non-zero: {'YES' if v else 'NO '}   "
                          f"= scheme(s) {match} up to a factor")
            # the trace is not identically zero
            assert tA, "the periplectic trace vanishes -- would contradict dim 3"
        # is A o B symmetric or antisymmetric?
        P = S.random_field(2, rng, nterms=3)
        Q = S.random_field(2, rng, nterms=3)
        ab = prod(S, P, Q)
        ba = prod(S, Q, P)
        rel = ("symmetric" if ab == ba else
               "antisymmetric" if ab == A.scale(ba, -1) else "neither")
        print(f"   A o B is {rel} in its two arguments")
        print(f"   parity of ptr(A): {A.parity(ptr(S, P))} (1 = odd)")


if __name__ == "__main__":
    main()
