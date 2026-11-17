#!/usr/bin/env python3
"""Can the connecting map H^0(quot) -> H^1(sub) of T_p kill the sub's class?

For E = V (x) d^k + sum d^{k_i} over LGr_n hitting the sub block of sub.py,
list the trivial constituents of H^0(X; Theta (x) Lambda^p E) (the only ones a
G-equivariant map can send onto a trivial H^1) and H^1 of the whole sub."""
import sys
from retract import (named, levi_weyl, weights, wedge, add_weight,
                     cohomology_of, direct_sum, retract, obstruction)
from bott import omega
from sub import dual, tensor


def sub_coh(X, E, p, W):
    q = wedge(E, p + 1)
    if not q:
        return {}
    return cohomology_of(X, tensor(dual(E), q), W)


def report(name, parts, p):
    X = named(name); W = levi_weyl(X)
    E = retract(X, parts)
    s = sub_coh(X, E, p, W)
    qq = obstruction(X, E, p, W)
    print("%s E=%s p=%d" % (name, parts, p))
    print("   H^0(quot) =", qq.get(0, []))
    print("   H^1(quot) =", qq.get(1, []))
    print("   H^1(sub)  =", s.get(1, []))
    print("   H^2(sub)  =", s.get(2, []))
    sys.stdout.flush()


if __name__ == "__main__":
    for n, k in ((3, 0), (3, 1), (3, -2), (5, 0)):
        X = named("LGr_%d" % n)
        r = X.R.rank
        idw = [0] * r; idw[0] = 1
        if n == 3:
            parts = [(idw, k), ([0] * r, -1 - k)]
            report("LGr_3", parts, 2)
        else:
            parts = [(idw, k), ([0] * r, -1 - 3 * k)]
            report("LGr_5", parts, 4)
