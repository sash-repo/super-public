"""The explicit shape of dv, the order-1 invariant operator metrics -> 1-forms.

There are two schemes of order 1 into 1-forms and only one combination of them is
invariant.  The two natural hand-written 1-forms are

    D(A)_i = omega^{jk} d_{z^j} A_{ki}          the "divergence" of A
    (d ptr A)_i = d_{z^i} (omega^{jk}A_{jk})    the differential of the trace

and this script finds the combination alpha D + beta d(ptr) that dv is, by exact
linear algebra on random fields -- so the text can print a formula instead of a
pair of scheme labels.

Usage:  python3 soddshapes.py [k]
"""

import random
import os
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace                             # noqa: E402

from ssuper import SSetup                                  # noqa: E402
from sodd0 import ptr                                      # noqa: E402
from sodddecomp import get_dv                              # noqa: E402


def naive_div(S, Atil):
    """omega^{jk} d_{z^j} d_{u^k} A~ -- the divergence, one u left free."""
    A = S.A
    return A.summ([A.scale(S.dz(S.du(Atil, j), i), w) for (i, j, w) in S.Omnz])


def d_ptr(S, Atil):
    """d(ptr A) = sum_i u^i d_{z^i}(omega^{jk}A_{jk})."""
    return S.free_deriv(ptr(S, Atil))


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    S = SSetup(k, k, form="odd")
    A = S.A
    dv, dvvec, dvsch = get_dv(S, k)
    print(f"PERIPLECTIC k = {k}: dv is the combination "
          f"{[str(c) for c in dvvec]} of the two schemes")

    rng = random.Random(31415 + k)
    rows, rhs = [], []
    for _ in range(4):
        P = S.random_field(3, rng, nterms=3)
        a, b, c = naive_div(S, P), d_ptr(S, P), dv(P)
        keys = sorted(set(a) | set(b) | set(c))
        for kk in keys:
            rows.append([a.get(kk, Fraction(0)), b.get(kk, Fraction(0)),
                         -c.get(kk, Fraction(0))])
    ns = nullspace(rows)
    sol = [v for v in ns if v[-1]]
    if not sol:
        print("dv is NOT a combination of the divergence and d(ptr A)")
        print(f"(nullspace dimension {len(ns)})")
        return
    v = sol[0]
    al, be = v[0] / v[2], v[1] / v[2]
    print(f"\n    dv A = ({al}) * omega^{{jk}} d_j A_{{ki}} u^i"
          f"  +  ({be}) * d_i(ptr A) u^i")
    if al:
        print(f"    i.e., up to the overall factor {al}:")
        print(f"        dv A = omega^{{jk}} d_j A_{{ki}} u^i"
              f"  +  ({be/al}) d_i(ptr A) u^i")

    # sanity: neither summand is invariant on its own
    print("\n  is either summand invariant on its own?")
    for name, F in (("omega^{jk} d_j A_{ki}", naive_div), ("d(ptr A)", d_ptr)):
        rows2 = []
        for _ in range(3):
            P = S.random_field(3, rng, nterms=3)
            x, y = F(S, P), dv(P)
            keys = sorted(set(x) | set(y))
            for kk in keys:
                rows2.append([x.get(kk, Fraction(0)), -y.get(kk, Fraction(0))])
        ns2 = nullspace(rows2)
        prop = any(w[1] for w in ns2)
        print(f"    {name:24s} proportional to dv: {'YES' if prop else 'NO'}")


if __name__ == "__main__":
    main()
