"""What the new operators are: the configuration S^2 x S^2 --> Lambda^2 over a
periplectic form, analysed.

  * the split of the space under the interchange A <--> B;
  * which of them are compositions -- the only unary invariants available are
    the identity, ptr and d(ptr), the census of sunfib.py showing that there is
    no unary invariant S^2 --> Lambda^2 at all;
  * whether d(ptr A)^d(ptr B) is one of them;
  * whether any scheme with an (out,out) pair, i.e. with a factor omega in the
    value, is ever invariant.

Usage:  python3 fibnamed.py [k] [N]
"""

import os
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank                                   # noqa: E402

import sfib                                                 # noqa: E402
from sfib import MSetup, enumerate_mixed, evaluate          # noqa: E402
from srunfib import analyse, operator                       # noqa: E402


def fingerprint(S, vec, schemes, invs, probes):
    A = S.A
    return [operator(S, vec, schemes, invs, P, Q) for (P, Q) in probes]


def flat(S, fps, keys):
    return [fp.get(k, Fraction(0)) for fp, kk in zip(fps, keys) for k in kk]


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    S = MSetup(k, k, shifts=(0, 0, 1), form="odd")
    A = S.A
    schemes, invs, ns, dim = analyse(S, N, verbose=True)
    print(f"S^2 x S^2 --> Lambda^2, k = {k}, order {N}: dimension {dim}")

    rng = random.Random(1234)
    probes = [(S.random_field(sfib.AFIB, max(3, N + 1), rng, nterms=3),
               S.random_field(sfib.BFIB, max(3, N + 1), rng, nterms=3))
              for _ in range(3)]
    fps = [fingerprint(S, v, schemes, invs, probes) for v in ns]
    keys = [sorted(set().union(*[set(fp[i]) for fp in fps])) if fps else []
            for i in range(len(probes))]

    def vecof(fp):
        return [fp[i].get(kk, Fraction(0)) for i in range(len(probes))
                for kk in keys[i]]

    M = [vecof(fp) for fp in fps]

    # 1) does any (out,out) scheme carry an invariant operator?
    oo = [i for i, s in enumerate(schemes) if s.inv is not None]
    print(f"\n  schemes with a factor omega in the value: {len(oo)} of "
          f"{len(schemes)}")
    if oo:
        # is the space spanned by the others alone?
        no_oo = [v for v in ns if not any(v[i] for i in oo)]
        Mn = [vecof(fingerprint(S, v, schemes, invs, probes)) for v in no_oo]
        d2 = rank(Mn) if Mn and Mn[0] else 0
        print(f"  the invariant operators that use none of them span {d2} of "
              f"{dim} -- so multiplying by omega gives "
              + ("nothing new" if d2 == dim else "SOMETHING"))

    # 2) the interchange A <--> B
    print("\n  under the interchange of the two arguments:")
    sym = anti = 0
    # T'(A,B) := T(B,A): the two arguments have the same symmetry type here,
    # so the fields are simply rewritten in the other family.
    swapped = []
    for v in ns:
        fp = [operator(S, v, schemes, invs,
                       S.reencode(Q, (sfib.BFIB, sfib.AFIB)),
                       S.reencode(P, (sfib.AFIB, sfib.BFIB)))
              for (P, Q) in probes]
        swapped.append(vecof(fp))
    base = rank(M)
    for sgn, name in ((1, "symmetric"), (-1, "antisymmetric")):
        rows = []
        for v, sw in zip(M, swapped):
            rows.append([a - sgn * b for a, b in zip(v, sw)])
        d = base - (rank(rows) if rows and rows[0] else 0)
        print(f"    {name}: {d}")
        if sgn == 1:
            sym = d
        else:
            anti = d
    print(f"    ({sym} + {anti} = {sym + anti} of {dim})")

    # 3) is d(ptr A) ^ d(ptr B) one of them, and is it alone?
    if N == 2:
        wedge = [A.mul(S.dform(S.ptr(P, sfib.AFIB)),
                       S.dform(S.ptr(Q, sfib.BFIB))) for (P, Q) in probes]
        w = vecof(wedge)
        print(f"\n  d(ptr A)^d(ptr B) lies in the space: "
              f"{rank(M + [w]) == base}")
        print(f"  ... and the space is {dim}-dimensional, so "
              f"{dim - 1} operator(s) beyond it")


if __name__ == "__main__":
    main()
