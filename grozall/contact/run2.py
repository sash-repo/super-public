"""k(1|2): the case with TWO weights.

Here k_0 = <K_t, K_{xi_1 xi_2}> is 2-dimensional abelian, so a density carries a pair
(lam, nu); the second multiplier is D_1D_2(f), derived in weights.py.  The target's
first weight is still lam1+lam2+W/2, but its second one is not determined by the
grading, so it is scanned: for each (nu1, nu2) a set of candidates nu3 is tried, and
the ones that carry an invariant operator are reported.
"""

import os, sys, time
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from gensuper import pencil, show_op                    # noqa: E402
from affine import dim_at, ops_at                       # noqa: E402
from runsuper import verify                             # noqa: E402

GEN = (Fr(17, 13), Fr(31, 11))


def scan(W, nus, cands, gdeg=3):
    n = 2
    out = []
    for (n1, n2) in nus:
        for c in cands:
            n3 = n1 + n2 + c
            terms, V0, N0, N1, N2 = pencil(n, W, rho2=(n1, n2, n3), gdeg=gdeg)
            if not terms:
                return
            d = dim_at(V0, N0, N1, N2, *GEN)
            if d:
                ops = ops_at(terms, V0, N0, N1, N2, *GEN)
                ok = all(verify(n, terms, v, W, GEN[0], GEN[1], (n1, n2, n3))
                         for v in ops)
                print("   nu = (%s, %s -> %s):  dim = %d at generic lam  [verified: %s]"
                      % (n1, n2, n3, d, ok))
                for v in ops:
                    print("        %s" % show_op(terms, v))
            out.append(((n1, n2, n3), d))
    return out


if __name__ == "__main__":
    NUS = [(Fr(0), Fr(0)), (Fr(1), Fr(0)), (Fr(1), Fr(-1)), (Fr(1, 2), Fr(1, 3))]
    CANDS = [Fr(0), Fr(1), Fr(-1), Fr(1, 2), Fr(-1, 2), Fr(2), Fr(-2)]
    for W in (0, 1, 2, 3):
        t0 = time.time()
        print("=" * 74)
        print("k(1|2), W = %d  (%s)" % (W, "odd" if W % 2 else "even"))
        scan(W, NUS, CANDS)
        print("   [%.0fs]" % (time.time() - t0))
        sys.stdout.flush()
