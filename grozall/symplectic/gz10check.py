"""Symbolic check of the printed formula for Gz_{1,0}.

What is checked, and how:

  1) the formula is evaluated by `iexpr.py`, which knows nothing of the
     scheme machinery of `gen.py`: it reads the very index expressions the
     text prints;
  2) at m = 2 the result is compared with the operator `rungen.py` returned,
     on fields the formula was NOT read off from;
  3) it is tested against EVERY Hamiltonian vector field whose generating
     function is a monomial of degree 1, 2 or 3 -- hence against
     h_{-1} + h_0 + h_1, which generates h(2m);
  4) its value is checked to lie in V_{(2,1)} at every point;
  5) and 3), 4) are repeated at m = 3, where the formula was not fitted,
     which is what makes it a formula uniform in m.

Usage:  python3 gz10check.py [--m 2] [--samples 2] [--nocompare]
"""

import argparse
import os
import pickle
import random
import sys
import time
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from symp import Symp, _monomials, padd, pscale  # noqa: E402
from fiber import Fiber, lie_tensor  # noqa: E402
from gen import Problem  # noqa: E402
from rungen import rand_field  # noqa: E402
from proj import Projector  # noqa: E402
import iexpr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = "gen_m2_2-1_2_2-1_N2.pkl"


def load_terms():
    """The twenty printed terms of Theta, from `gz10terms.py`."""
    from gz10terms import TERMS
    return list(TERMS), None


def apply_formula(S, pi, terms, chi, A):
    return pi.apply_field(iexpr.evaluate(S, terms, chi, A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--fdeg", type=int, default=3)
    ap.add_argument("--nterms", type=int, default=2)
    ap.add_argument("--degs", default="1,2,3")
    ap.add_argument("--nocompare", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    terms, D = load_terms()
    print(f"formula with {len(terms)} terms, m = {args.m}")
    S = Symp(args.m)
    Fl, Fn = Fiber(S, (2, 1)), Fiber(S, (2, 1))
    Fm = Fiber(S, (2,))
    pi = Projector(S, Fn)
    print(f"fibers built (sdim V = {Fn.dim})  [{time.time()-t0:.1f}s]")

    rng = random.Random(args.seed)
    hams = [({e: Fraction(1)}, d) for d in [int(x) for x in args.degs.split(",")]
            for e in _monomials(S.n, d)]

    # the engine's answer, for the comparison at m = 2
    eng = None
    if args.m == 2 and not args.nocompare:
        with open(os.path.join(HERE, PKL), "rb") as fh:
            G = pickle.load(fh)
        P = Problem(S, Fl, Fm, Fn, G["order"])
        ker = [[Fraction(c) for c in v] for v in G["kernel"]]

        def engine(A, B):
            vals = P.eval_all(A, B)          # once: the expensive part
            for v in ker:
                acc = [dict() for _ in range(S.n ** 3)]
                for c, V in zip(v, vals):
                    if c:
                        for i in range(S.n ** 3):
                            if V[i]:
                                acc[i] = padd(acc[i], pscale(V[i], c))
                if any(acc):
                    return acc
            return None
        eng = engine

    bad, nonzero, ratio = [], False, None
    for s in range(args.samples):
        chi = rand_field(Fl, rng, args.fdeg, args.nterms)
        A = rand_field(Fm, rng, args.fdeg, args.nterms)
        R = apply_formula(S, pi, terms, chi, A)
        if any(R):
            nonzero = True
        # (a) the value is a tensor field of type (2,1)
        mons = set()
        for i in range(S.n ** 3):
            mons |= set(R[i])
        for e in mons:
            val = [R[i].get(e, Fraction(0)) for i in range(S.n ** 3)]
            if not Fn.contains(val):
                bad.append(("membership", s, e))
        # (b) equal to the engine's operator, up to one global factor
        if eng is not None:
            E = eng(chi, A)
            r = None
            for i in range(S.n ** 3):
                for e, c in E[i].items():
                    if c:
                        r = R[i].get(e, Fraction(0)) / c
                        break
                if r is not None:
                    break
            if ratio is None:
                ratio = r
            if r != ratio:
                bad.append(("scale", s, r, ratio))
            for i in range(S.n ** 3):
                d = padd(R[i], pscale(E[i], -ratio))
                if d:
                    bad.append(("engine", s, i))
                    break
        # (c) equivariance
        for f, d in hams:
            Lc = lie_tensor(S, 3, f, chi)
            LA = lie_tensor(S, 2, f, A)
            LR = lie_tensor(S, 3, f, R)
            R1 = apply_formula(S, pi, terms, Lc, A)
            R2 = apply_formula(S, pi, terms, chi, LA)
            for i in range(S.n ** 3):
                dd = padd(LR[i], pscale(padd(R1[i], R2[i]), -1))
                if dd:
                    bad.append(("equivariance", s, d, i))
                    break
        print(f"  sample {s+1}: {len(hams)} Hamiltonians tested; "
              f"defects so far {len(bad)}  [{time.time()-t0:.1f}s]")

    print()
    print(f"nonzero: {nonzero}")
    if ratio is not None:
        print(f"formula = {ratio} x (the engine's operator)")
    print(f"VERDICT: {'PASS' if nonzero and not bad else 'FAIL'}"
          + ("" if not bad else f"  ({len(bad)} defects, first {bad[0]})"))
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
