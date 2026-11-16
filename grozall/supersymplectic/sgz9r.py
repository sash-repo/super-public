"""Formula (9), superized, against the coefficient vector read off by suniv.py.

sblocks.py shows that the printed formula (9), with its blocks read in the
encoding (uenc), is invariant and non-zero on every symplectic supermanifold
tried.  Since the space of such operators is 1-dimensional (Statement
10.7 of the text), that already identifies it with Gz(2m|n).  This program computes
the factor, which is the quantitative form of the same statement and a check on
both computations at once: the two expressions were built by different routes --
one from Grozman's blocks, one from a null space of the scheme ansatz -- and
must be proportional.

Usage:  python3 sgz9r.py [m n]...
"""

import os
import pickle
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from ssuper import SSetup                                  # noqa: E402
from sgen import enumerate_schemes, evaluate               # noqa: E402
from sblocks import BLOCKS, evaluate_block, ratio, fit_constant  # noqa: E402
from suniv import canonical, zero_subspace                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def constants():
    """The constants of the encoding, fixed at n = 0 as in sblocks.py."""
    S = SSetup(2, 0)
    rng = random.Random(4244)
    pairs = [(S.random_field(3, rng, nterms=3), S.random_field(3, rng, nterms=3))
             for _ in range(2)]
    out = {}
    for (name, fun, kinds, FA, FB, c) in BLOCKS:
        out[name] = fit_constant(S, kinds, FA, FB, fun, pairs)
    return out


def gz9(S, kappa, P, Q):
    return S.A.summ([
        S.A.scale(evaluate_block(S, kinds, FA, FB, P, Q), c * kappa[name])
        for (name, fun, kinds, FA, FB, c) in BLOCKS])


def main():
    args = [int(x) for x in sys.argv[1:]]
    cases = ([(args[i], args[i + 1]) for i in range(0, len(args), 2)]
             or [(1, 0), (2, 0), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2)])
    kappa = constants()
    print("constants of the encoding:",
          ", ".join(f"{k}={v}" for k, v in kappa.items()))
    schemes = enumerate_schemes(2)

    # the reference vector of suniv.py, read off at (2|3)
    S0 = SSetup(1, 3)
    with open(os.path.join(HERE, "sgen_even_m1_n3_N2.pkl"), "rb") as fh:
        D = pickle.load(fh)
    kernel = [[Fraction(c) for c in v] for v in D["kernel"]]
    Z = zero_subspace(S0, schemes, kernel, random.Random(31337))
    v = canonical(kernel, Z)
    print(f"the read-off vector: {sum(1 for c in v if c)} non-zero coefficients\n")

    for (m, n) in cases:
        S = SSetup(m, n)
        rng = random.Random(1234 + 7 * m + n)
        ok = True
        rs = set()
        for _ in range(2):
            P = S.random_field(4, rng, nterms=3)
            Q = S.random_field(4, rng, nterms=3)
            lhs = gz9(S, kappa, P, Q)
            rhs = S.A.summ([S.A.scale(evaluate(S, sch, P, Q), c)
                            for c, sch in zip(v, schemes) if c])
            if not lhs and not rhs:
                continue
            r = ratio(lhs, rhs)
            if r is None:
                ok = False
            else:
                rs.add(r)
        print(f"  (2m|n) = ({2*m}|{n}):  "
              + ("NOT PROPORTIONAL" if not ok else
                 f"formula (9) = {' or '.join(str(x) for x in sorted(rs))}"
                 " x the read-off operator"))


if __name__ == "__main__":
    main()
