"""In what sense H(2:n;omega_2) is a filtered deformation of the graded one.

With the standard degree deg x^(i)y^(j) = i+j-2 of the Poisson model, the
omega_2 bracket differs from the omega_0 bracket in exactly one place: when
the omega_0 bracket of two elements is a nonzero constant, the factor (1+e)
puts that constant back on the top monomial e instead.  This script checks
that this happens only for pairs of elements of degree -1, so that the extra
term always raises the degree, i.e., the bracket preserves the standard
filtration and the deformation sits at the very top of it.
"""
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from nongraded import Ham                                     # noqa: E402


def check(p, n1, n2):
    L2, L0 = Ham(p, n1, n2, 2), Ham(p, n1, n2, 0)
    L2.table()
    L0.table()
    deg = lambda m: m[0] + m[1] - 2                            # noqa: E731
    top = L2.index[L2.top]
    deg_e = deg(L2.top)
    extra, bad, mismatch = 0, 0, 0
    for a in range(L2.dim):
        for b in range(L2.dim):
            ma, mb = L2.basis[a], L2.basis[b]
            br2 = dict(L2._table[a][b])
            c = br2.pop(top, 0)
            # the omega_0 bracket of the same two monomials, when both are in L0
            if ma in L0.index and mb in L0.index:
                br0 = {L2.index[L0.basis[k]]: v
                       for k, v in L0._table[L0.index[ma]][L0.index[mb]].items()}
                b2 = dict(br2)
                b2.pop(top, None)
                if {k: v for k, v in b2.items() if v} != {k: v for k, v in br0.items() if v}:
                    # they may differ exactly by a term on e, allowed
                    mismatch += 1
            if c and (ma, mb) != (L2.top, L2.top):
                extra += 1
                if not (deg(ma) == -1 and deg(mb) == -1):
                    bad += 1
    print("p=%d n=(%d,%d): deg e = %d;  extra e-terms: %d, of them outside "
          "[L_{-1},L_{-1}]: %d;  bracket mismatches off e: %d"
          % (p, n1, n2, deg_e, extra, bad, mismatch))


if __name__ == "__main__":
    for case in [(2, 2, 2), (3, 1, 1), (3, 1, 2), (5, 1, 1), (7, 1, 1)]:
        check(*case)
