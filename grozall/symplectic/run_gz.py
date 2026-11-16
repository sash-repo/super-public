"""Classify the Diff_omega-invariant bilinear operators

        T(2,0,...,0) x T(2,0,...,0) --> T(2,0,...,0)

of order N on a 2m-dimensional symplectic manifold, i.e. the operators on
covariant symmetric 2-tensor fields ("metrics").  Grozman's operator Gz is the
case N = 2; he printed it only for m = 1 and asked for m > 1.

Usage:  python3 run_gz.py [m] [N] [nsamples]
"""

import sys, os, random
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank, backend  # noqa: E402

from symp import (Symp, enumerate_terms, eval_term_sym, random_sym_field,  # noqa: E402
                  monomial_polys, padd, pscale, piszero, _sum_polys)


def flatten(S, T, index):
    """Flatten a symmetric tensor of polynomials into a coefficient dict."""
    out = {}
    n = S.n
    for i in range(n):
        for j in range(i, n):
            for e, c in T[i][j].items():
                out[(i, j, e)] = out.get((i, j, e), Fraction(0)) + c
    return out


def defect(S, term, f, A, B):
    """L_f(T(A,B)) - T(L_f A, B) - T(A, L_f B)."""
    TAB = eval_term_sym(S, term, A, B)
    lhs = S.lie_sym2(f, TAB)
    LA = S.lie_sym2(f, A)
    LB = S.lie_sym2(f, B)
    r1 = eval_term_sym(S, term, LA, B)
    r2 = eval_term_sym(S, term, A, LB)
    n = S.n
    out = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = padd(lhs[i][j], pscale(padd(r1[i][j], r2[i][j]), -1))
            out[i][j] = s
            out[j][i] = s
    return out


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    nsamp = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    S = Symp(m)
    terms = enumerate_terms(N)
    print(f"m = {m}, n = {S.n}, order N = {N}, backend = {backend()}")
    print(f"ansatz terms: {len(terms)}")

    rng = random.Random(20260830 + m * 100 + N)
    ncub = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    fdeg = int(sys.argv[5]) if len(sys.argv) > 5 else N + 2
    cubics = monomial_polys(S, 3)
    if ncub:
        # h_1 = S^3(W) is an irreducible sp(2m)-module, so ONE cubic whose
        # sp-orbit spans it suffices once the ansatz is Sp-covariant (it is, by
        # construction).  x_1^3 is a weight vector generating S^3(W).
        cubics = cubics[:ncub]
    print(f"cubic Hamiltonians tested: {len(cubics)}")

    rows = []          # each row: list of coefficients, one per term
    colkeys = {}

    def addblock(defects):
        """defects: list (per term) of symmetric tensors; make equations."""
        keys = set()
        for D in defects:
            for i in range(S.n):
                for j in range(i, S.n):
                    for e in D[i][j]:
                        keys.add((i, j, e))
        for (i, j, e) in sorted(keys):
            rows.append([D[i][j].get(e, Fraction(0)) for D in defects])

    for s in range(nsamp):
        A = random_sym_field(S, fdeg, rng)
        B = random_sym_field(S, fdeg, rng)
        for fi, f in enumerate(cubics):
            defects = [defect(S, t, f, A, B) for t in terms]
            addblock(defects)
        print(f"  sample {s+1}: {len(rows)} equations so far")

    if not rows:
        print("no equations")
        return
    ns = nullspace(rows)
    print(f"coefficient-space kernel dimension: {len(ns)} (of {len(terms)} terms)")

    # the kernel may contain vectors that give the *zero* operator (linear
    # relations among the ansatz terms); measure the dimension of the space of
    # actual operators.
    rng2 = random.Random(777)
    pdeg = max(3, N + 1)
    probes = [(random_sym_field(S, pdeg, rng2), random_sym_field(S, pdeg, rng2)) for _ in range(2)]
    fps = []
    for v in ns:
        fp = []
        for (A, B) in probes:
            acc = [[{} for _ in range(S.n)] for _ in range(S.n)]
            for c, t in zip(v, terms):
                if c == 0:
                    continue
                T = eval_term_sym(S, t, A, B)
                for i in range(S.n):
                    for j in range(i, S.n):
                        acc[i][j] = padd(acc[i][j], pscale(T[i][j], c))
            fp.append(acc)
        fps.append(fp)
    kk = []
    for pi in range(len(probes)):
        s = set()
        for fp in fps:
            acc = fp[pi]
            for i in range(S.n):
                for j in range(i, S.n):
                    s |= set((i, j, e) for e in acc[i][j])
        kk.append(sorted(s))
    M = []
    for fp in fps:
        row = []
        for pi in range(len(probes)):
            acc = fp[pi]
            for (i, j, e) in kk[pi]:
                row.append(acc[i][j].get(e, Fraction(0)))
        M.append(row)
    r = rank(M) if M and M[0] else 0
    print(f"dimension of the space of INVARIANT OPERATORS of order {N}: {r}")

    # zero-operator relations
    print(f"(relations among ansatz terms inside the kernel: {len(ns) - r})")



if __name__ == "__main__":
    main()
