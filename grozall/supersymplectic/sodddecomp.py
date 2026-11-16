"""Are the three order-2 operators over an ODD symplectic form COMPOSITIONS?

The unary invariant operators on metrics are, by soddcensus.py, exactly three:

        id                              order 0, even, metrics -> metrics
        ptr(A) = omega^{ij}A_{ij}       order 0, ODD,  metrics -> functions
        dv A                            order 1, ODD,  metrics -> 1-forms

and there is NO invariant unary operator of order 2 on metrics at all -- neither
into metrics (3 schemes, kernel 0) nor into functions (4 schemes, kernel 3, every
combination of them the zero operator).  So the composites of total order 2 are
short to list:

  C1(A,B) = L_{X_{ptr A}} B      ptr, then the Lie derivative along the
  C2(A,B) = L_{X_{ptr B}} A      Hamiltonian field of the resulting function
  C3(A,B) = dvA . dvB            the product of the two 1-forms

Each of the three is invariant BY CONSTRUCTION -- a composition of equivariant
maps -- so the job of the arithmetic is not to re-prove that but to say WHICH
element of the 3-dimensional space each one is, and whether together they exhaust
it.  (A naive Leibniz test on these hand-built formulas is not the way to check:
it feeds them tensor fields of ODD parity, where a hand-written formula and the
scheme evaluation differ by a Koszul sign, and it then reports a false negative.
This is the second sign trap of Section 11 of the text.)

dv is NOT guessed: there are two schemes of order 1 into 1-forms and the
invariant operator is one particular combination of them, taken here from
sunary.invariant_unary.  Guessing one of the two schemes gives an operator whose
square-product is not invariant at all -- it lands outside the 3-space.

Usage:  python3 sodddecomp.py [k]
"""

import os
import pickle
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank, nullspace                       # noqa: E402

from ssuper import SSetup                                  # noqa: E402
from sgen import enumerate_schemes, evaluate               # noqa: E402
from suniv import zero_subspace, rref                      # noqa: E402
from sodd0 import ptr                                      # noqa: E402
from sunary import invariant_unary, evaluate_unary         # noqa: E402

N = 2


def get_dv(S, k):
    """The invariant unary operator of order 1 into 1-forms, from the census."""
    rng = random.Random(9110 + k)
    schemes, ns, dim = invariant_unary(S, 1, 1, rng, verbose=False)
    assert dim == 1, f"expected a 1-dimensional space of order-1 1-forms, got {dim}"
    v = next(w for w in ns if any(w))
    A = S.A

    def dv(X):
        return A.summ([A.scale(evaluate_unary(S, s, X), c)
                       for c, s in zip(v, schemes) if c])

    return dv, v, schemes


def composites(S, dv):
    A = S.A
    return [("C1 = L_{X_{ptr A}} B", lambda P, Q: S.lie(ptr(S, P), Q)),
            ("C2 = L_{X_{ptr B}} A", lambda P, Q: S.lie(ptr(S, Q), P)),
            ("C3 = dvA . dvB      ", lambda P, Q: A.mul(dv(P), dv(Q)))]


def symmetry(S, T, rng, fdeg=3):
    A = S.A
    P = S.random_field(fdeg, rng, nterms=3)
    Q = S.random_field(fdeg, rng, nterms=3)
    x, y = T(P, Q), T(Q, P)
    if not x and not y:
        return "zero"
    if x == y:
        return "symmetric"
    if x == A.scale(y, -1):
        return "antisymmetric"
    return "neither"


def basis_order2(S, k, schemes):
    with open(f"sgen_odd_m{k}_n{k}_N{N}.pkl", "rb") as fh:
        D = pickle.load(fh)
    kernel = [[Fraction(c) for c in v] for v in D["kernel"]]
    Z = zero_subspace(S, schemes, kernel, random.Random(31337))
    ZR, piv = rref(Z)
    reps = []
    for v in kernel:
        w = list(v)
        for row, c in zip(ZR, piv):
            if w[c]:
                f = w[c]
                w = [a - f * b for a, b in zip(w, row)]
        if any(w):
            reps.append(w)
    R, _ = rref(reps)
    return R, D["dim"]


def flatten(vals, keysets):
    out = []
    for v, keys in zip(vals, keysets):
        for kk in keys:
            out.append(v.get(kk, Fraction(0)))
    return out


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    nprobe = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    S = SSetup(k, k, form="odd")
    A = S.A
    schemes = enumerate_schemes(N)
    print(f"PERIPLECTIC k = {k}: sdim ({k}|{k}), le({k}), aut(omega) = pe({k})")

    R, dim = basis_order2(S, k, schemes)
    print(f"the space of invariant order-2 operators: dimension {dim}, "
          f"{len(R)} independent representatives among {len(schemes)} schemes")

    dv, dvvec, dvsch = get_dv(S, k)
    print(f"dv, the invariant order-1 operator into 1-forms, is the combination "
          f"{[str(c) for c in dvvec]} of")
    for s in dvsch:
        print(f"      {s}")

    comps = composites(S, dv)
    print("\n  symmetry of each composite in its two arguments:")
    for label, T in comps:
        print(f"    {label}  {symmetry(S, T, random.Random(99 + k))}")

    rng2 = random.Random(20260904 + k)
    probes = [(S.random_field(4, rng2, nterms=3), S.random_field(4, rng2, nterms=3))
              for _ in range(nprobe)]

    def opvals(v):
        return [A.summ([A.scale(evaluate(S, sch, P, Q), c)
                        for c, sch in zip(v, schemes) if c]) for (P, Q) in probes]

    basisvals = [opvals(v) for v in R]
    compvals = [[T(P, Q) for (P, Q) in probes] for _, T in comps]
    keysets = [sorted(set().union(*[set(x[i]) for x in basisvals + compvals]))
               for i in range(len(probes))]
    Bf = [flatten(x, keysets) for x in basisvals]
    Cf = [flatten(x, keysets) for x in compvals]

    print(f"\n  probes: {nprobe} pairs of random fields of degree 4;"
          f" {len(Bf[0])} scalar equations each")
    print(f"  rank of the basis                   : {rank(Bf)}")
    print(f"  rank of the three composites        : {rank(Cf)}")
    print(f"  rank of basis + composites together : {rank(Bf + Cf)}"
          f"   (= {rank(Bf)} <=> every composite lies in the space)")

    print("\n  each composite, in the basis of the space:")
    inside = 0
    for (label, _), c in zip(comps, Cf):
        M = [[Bf[i][j] for i in range(len(Bf))] + [-c[j]] for j in range(len(c))]
        ns = nullspace(M)
        sol = [v for v in ns if v[-1]]
        if not sol:
            print(f"    {label}  NOT in the span of the basis (!)")
            continue
        inside += 1
        v = sol[0]
        coeff = [x / v[-1] for x in v[:-1]]
        print(f"    {label}  = " + " + ".join(
            f"({c2}) e{i+1}" for i, c2 in enumerate(coeff) if c2))

    print(f"\n  composites lying in the space: {inside} of {len(comps)}")
    print(f"  DIMENSION SPANNED BY THE COMPOSITES: {rank(Cf)} of {dim}")
    if inside == len(comps) and rank(Cf) == dim:
        print("  ==> the space is spanned by compositions: over an odd form there"
              " is NO new operator of order 2.")
    else:
        print(f"  ==> {dim - rank(Cf)} dimension(s) not accounted for.")


if __name__ == "__main__":
    main()
