"""Independent verification of an operator found by srun.py.

The operator is taken back from the saved kernel and tested on FRESH random
fields it was not built from, against EVERY monomial Hamiltonian of degree
1, 2 and 3 -- that is, against all of h_{-1} + h_0 + h_1, which generates
h(2m|n) (checked in test_ssuper.py).  Nothing of the scheme machinery is
trusted: the test is the invariance identity itself,

        L_f(T(A,B)) = T(L_f A, B) + T(A, L_f B),

with the Lie derivative that the calibration of test_ssuper.py fixed.

Usage:  python3 sverify.py m n N
"""

import os
import pickle
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank                                  # noqa: E402

from ssuper import SSetup                                  # noqa: E402
from sgen import Scheme, evaluate                          # noqa: E402
from srun import op_parity                                 # noqa: E402


def load(m, n, N, form):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"sgen_{form}_m{m}_n{n}_N{N}.pkl"), "rb") as fh:
        return pickle.load(fh)


def main():
    m, n, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    form = sys.argv[4] if len(sys.argv) > 4 else "even"
    D = load(m, n, N, form)
    S = SSetup(m, n, form=form)
    schemes = [Scheme(*s) for s in D["schemes"]]
    kernel = [[Fraction(c) for c in v] for v in D["kernel"]]
    print(f"(m,n) = ({m},{n}), order {N}: {len(kernel)} kernel vectors, "
          f"operator dimension {D['dim']}")
    if not kernel:
        print("nothing to verify")
        return

    pT = op_parity(S, N)
    rng = random.Random(999 + m * 31 + n * 7 + N)
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
    A = S.A
    nonzero = 0
    bad = 0
    for vi, v in enumerate(kernel):
        ok = True
        seen_nonzero = False
        for _ in range(2):
            Atil = S.random_field(N + 2, rng, nterms=3)
            Btil = S.random_field(N + 2, rng, nterms=3)

            def T(P, Q):
                return A.summ([A.scale(evaluate(S, sch, P, Q), c)
                               for c, sch in zip(v, schemes) if c])

            val = T(Atil, Btil)
            if val:
                seen_nonzero = True
            for f in hams:
                pX = (A.parity(f) + (1 if S.form == "odd" else 0)) % 2
                e = -1 if (pT * pX) % 2 else 1
                lhs = S.lie(f, val)
                rhs = A.scale(A.add(T(S.lie(f, Atil), Btil),
                                    T(Atil, S.lie(f, Btil))), e)
                if lhs != rhs:
                    ok = False
                    print(f"  kernel vector {vi}: FAILS on a Hamiltonian")
                    break
            if not ok:
                break
        if seen_nonzero:
            nonzero += 1
        if not ok:
            bad += 1
    print(f"kernel vectors that are non-zero as operators (on the probes): {nonzero}")
    print(f"kernel vectors failing the invariance identity: {bad}")
    print("VERDICT:", "PASS" if bad == 0 else "FAIL")

    # and the value must lie in the fiber: quadratic in u, i.e. every monomial
    # of the answer has u-degree exactly 2
    Atil = S.random_field(N + 2, rng, nterms=3)
    Btil = S.random_field(N + 2, rng, nterms=3)
    uev = set(S.uev.values())
    uod = set(S.uod.values())
    okdeg = True
    for v in kernel:
        val = A.summ([A.scale(evaluate(S, sch, Atil, Btil), c)
                      for c, sch in zip(v, schemes) if c])
        for (e, msk) in val:
            d = sum(e[i] for i in uev) + sum(1 for a in uod if (msk >> a) & 1)
            if d != 2:
                okdeg = False
    print("every value lies in the fiber (u-degree 2):", "PASS" if okdeg else "FAIL")


if __name__ == "__main__":
    main()
