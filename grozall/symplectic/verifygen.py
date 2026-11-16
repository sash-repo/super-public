"""Independent verification of an operator produced by `rungen.py`.

The solve uses ONE cubic Hamiltonian (legitimate: h_1 = S^3(W) is an irreducible
sp(2m)-module and the ansatz is Sp-covariant).  This script takes the answer and
checks it the hard way:

  * on FRESH random fields, not the ones the system was built from;
  * against EVERY monomial cubic Hamiltonian, not one;
  * against quadratic and linear Hamiltonians as well, so that h_{-1} + h_0 +
    h_1 -- which generates the whole of h(2m) -- is covered;
  * and it re-checks that the value lies in V_nu.

Usage:  python3 verifygen.py gen_<tag>.pkl [--samples 3] [--seed 7]
"""

import argparse
import os
import pickle
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from symp import Symp, _monomials, padd, pscale  # noqa: E402
from fiber import Fiber, lie_tensor  # noqa: E402
from gen import Problem  # noqa: E402
from rungen import rand_field  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def hamiltonians(S, degs):
    out = []
    for d in degs:
        for e in _monomials(S.n, d):
            out.append(({e: Fraction(1)}, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fdeg", type=int, default=3)
    ap.add_argument("--nterms", type=int, default=3)
    args = ap.parse_args()

    with open(os.path.join(HERE, args.pkl), "rb") as fh:
        D = pickle.load(fh)
    S = Symp(D["m"])
    Fl, Fm, Fn = Fiber(S, D["lam"]), Fiber(S, D["mu"]), Fiber(S, D["nu"])
    P = Problem(S, Fl, Fm, Fn, D["order"])
    ker = [[Fraction(c) for c in v] for v in D["kernel"]]
    print(f"m = {D['m']},  T{D['lam']} x T{D['mu']} --> T{D['nu']},  order {D['order']}")
    print(f"kernel {len(ker)} vectors over {len(P.terms)} schemes; "
          f"reported operator dimension {D['opdim']}")
    if D["opdim"] == 0:
        print("nothing to verify: the operator space is trivial")
        return

    rng = random.Random(args.seed)
    # a generic member of the solution space
    coef = [Fraction(0)] * len(P.terms)
    for v in ker:
        c = rng.randint(-9, 9)
        if c:
            for i, x in enumerate(v):
                if x:
                    coef[i] += c * x

    def apply(A, B):
        vals = P.eval_all(A, B)
        acc = [dict() for _ in range(S.n ** Fn.k)]
        for c, V in zip(coef, vals):
            if c == 0:
                continue
            for i in range(S.n ** Fn.k):
                if V[i]:
                    acc[i] = padd(acc[i], pscale(V[i], c))
        return acc

    hams = hamiltonians(S, [1, 2, 3])
    nonzero = False
    bad = []
    for s in range(args.samples):
        A = rand_field(Fl, rng, args.fdeg, args.nterms)
        B = rand_field(Fm, rng, args.fdeg, args.nterms)
        R = apply(A, B)
        if any(R[i] for i in range(S.n ** Fn.k)):
            nonzero = True
        # (a) the value is a tensor field of type nu
        mons = set()
        for i in range(S.n ** Fn.k):
            mons |= set(R[i])
        for e in mons:
            val = [R[i].get(e, Fraction(0)) for i in range(S.n ** Fn.k)]
            if not Fn.contains(val):
                bad.append(("membership", s, e))
        # (b) equivariance
        for f, d in hams:
            LA = lie_tensor(S, Fl.k, f, A)
            LB = lie_tensor(S, Fm.k, f, B)
            LR = lie_tensor(S, Fn.k, f, R)
            R1 = apply(LA, B)
            R2 = apply(A, LB)
            for i in range(S.n ** Fn.k):
                dd = padd(LR[i], pscale(padd(R1[i], R2[i]), -1))
                if dd:
                    bad.append(("equivariance", s, d, i))
        print(f"  sample {s+1}: checked {len(hams)} Hamiltonians "
              f"(degrees 1,2,3); defects so far {len(bad)}")

    print()
    print(f"operator is nonzero: {nonzero}")
    print(f"VERDICT: {'PASS' if (nonzero and not bad) else 'FAIL'}"
          + ("" if not bad else f"  ({len(bad)} defects, first: {bad[0]})"))


if __name__ == "__main__":
    main()
