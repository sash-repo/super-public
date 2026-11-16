"""Is the super Grozman operator ONE formula, uniform in the superdimension?

srun.py says the space of invariant order-2 operators on metrics is
1-dimensional for every (2m|n) tried.  That alone does not say the operators
form one family: a priori each superdimension could need its own combination of
contraction schemes, as the four blocks of Grozman's m = 1 formula famously fail
to span anything for m > 1.

Here the coefficient vector is read off in ONE case, reduced to a canonical
representative modulo the combinations that give the zero operator, and then
tested, unchanged, in all the others.

Usage:  python3 suniv.py [refm refn] -- default reference (m,n) = (1,3)
"""

import os
import pickle
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank                       # noqa: E402

from ssuper import SSetup                                  # noqa: E402
from sgen import Scheme, enumerate_schemes, evaluate       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 2


def rref(rows):
    """Row-reduce over Q; returns (reduced rows, pivot columns)."""
    M = [list(r) for r in rows]
    piv = []
    r = 0
    if not M:
        return [], []
    ncol = len(M[0])
    for c in range(ncol):
        pr = None
        for i in range(r, len(M)):
            if M[i][c]:
                pr = i
                break
        if pr is None:
            continue
        M[r], M[pr] = M[pr], M[r]
        inv = Fraction(1) / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    return M[:r], piv


def zero_subspace(S, schemes, kernel, rng, nprobe=3):
    """The kernel vectors that give the ZERO operator."""
    A = S.A
    probes = [(S.random_field(3, rng, nterms=3), S.random_field(3, rng, nterms=3))
              for _ in range(nprobe)]
    vals = []
    for (P, Q) in probes:
        vals.append([evaluate(S, sch, P, Q) for sch in schemes])
    keys = []
    for v in vals:
        keys.append(sorted(set().union(*[set(x) for x in v]) if v else set()))
    rows = []
    for pi, v in enumerate(vals):
        for k in keys[pi]:
            rows.append([x.get(k, Fraction(0)) for x in v])
    return nullspace(rows) if rows else []


def canonical(kernel, zero):
    """A representative of kernel/zero, made sparse and normalised."""
    Z, piv = rref(zero)
    for v in kernel:
        w = list(v)
        for row, c in zip(Z, piv):
            if w[c]:
                f = w[c]
                w = [a - f * b for a, b in zip(w, row)]
        if any(w):
            first = next(x for x in w if x)
            return [x / first for x in w]
    return None


def test_vector(S, schemes, v, rng, label):
    A = S.A
    ok, nonzero = True, False
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
    for _ in range(2):
        P = S.random_field(4, rng, nterms=3)
        Q = S.random_field(4, rng, nterms=3)

        def T(X, Y):
            return A.summ([A.scale(evaluate(S, sch, X, Y), c)
                           for c, sch in zip(v, schemes) if c])

        val = T(P, Q)
        if val:
            nonzero = True
        for f in hams:
            if S.lie(f, val) != A.add(T(S.lie(f, P), Q), T(P, S.lie(f, Q))):
                ok = False
                break
        if not ok:
            break
    print(f"  {label:28s} invariant: {'YES' if ok else 'NO '}   "
          f"non-zero: {'YES' if nonzero else 'NO '}")
    return ok and nonzero


def main():
    refm = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    refn = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    schemes = enumerate_schemes(N)
    with open(os.path.join(HERE, f"sgen_even_m{refm}_n{refn}_N{N}.pkl"), "rb") as fh:
        D = pickle.load(fh)
    kernel = [[Fraction(c) for c in v] for v in D["kernel"]]
    S = SSetup(refm, refn)
    rng = random.Random(31337)
    Z = zero_subspace(S, schemes, kernel, rng)
    print(f"reference (m,n) = ({refm},{refn}): kernel {len(kernel)}, "
          f"zero-operator subspace {len(Z)}, quotient {len(kernel)-len(Z)}")
    v = canonical(kernel, Z)
    if v is None:
        print("no non-zero operator in the kernel")
        return
    nz = [(i, c) for i, c in enumerate(v) if c]
    print(f"canonical representative: {len(nz)} non-zero coefficients")
    for i, c in nz:
        print(f"    {str(c):>8s}  *  {schemes[i]}")

    print("\nthe SAME vector, tested in every other superdimension:")
    allok = True
    for (m, n) in [(1, 0), (2, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                   (2, 1), (2, 2), (3, 0), (2, 3)]:
        S2 = SSetup(m, n)
        allok &= test_vector(S2, schemes, v, random.Random(77 + 5 * m + n),
                             f"(2m|n) = ({2*m}|{n})")
    print("\nUNIFORM IN THE SUPERDIMENSION:", "YES" if allok else "NO")


if __name__ == "__main__":
    main()
