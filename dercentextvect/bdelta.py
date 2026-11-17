#!/usr/bin/env python3
"""Is the odd Laplacian a derivation of the member b_0(n) = b(n) of the family?

b(n) = {M_f in m(n) | d_tau f = 0}: on such f the pericontact bracket is minus
the Buttin bracket, so b(n) is the antibracket algebra on the whole of K[q, xi],
constants included -- whence its center K M_1 and the exact sequence
0 -> K M_1 -> b(n) -> le(n) -> 0.  The odd Laplacian

    Delta = sum_i d_{q_i} d_{xi_i}

generates the antibracket, so it should be a derivation here exactly as it is of
le(n).  This checks that, and whether it is induced by Der O.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dervect import Echelon                                        # noqa: E402
from laplace import test_sum, op_on_mon                            # noqa: E402
import mvect                                                       # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bdelta.log")


def main():
    with open(LOG, "w") as fh:
        for p, n in ((3, 1), (5, 1), (3, 2)):
            S = mvect.make(p, n)
            M, fields = mvect.pericontact(S, n)
            B = mvect.buttin(S, n, 0, 1, fields)
            keep = {u: X for u, X in fields.items()
                    if Echelon([S.vec(Y) for Y in B], S.p).contains(S.vec(X))}

            def delta(u, S=S, n=n):
                out = {}
                for i in range(n):
                    for w, c in op_on_mon(
                            S, [('x', i, 0), ('xi', 1 + i)], u).items():
                        out[w] = (out.get(w, 0) + c) % S.p
                return {k: v for k, v in out.items() if v}

            test_sum(f"p={p} b_0({n}) = b({n}), Delta", S, keep, delta, fh)


if __name__ == "__main__":
    main()
