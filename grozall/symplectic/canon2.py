"""Same read-off as canon.py, but with a curated ordering of the candidate
operations: Grozman's four m = 1 blocks first, then the manifestly
A <-> A' symmetric groups, then the raw contractions.  The greedy choice of an
independent family therefore uses the recognisable operations first and the
answer comes out as short as this basis allows.

Usage:  python3 canon2.py [m] [nsamples] [nterms-per-entry]
"""

import sys, os, random
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank  # noqa: E402

from symp import (Symp, enumerate_terms, eval_term_sym, random_sym_field,  # noqa: E402
                  monomial_polys, padd, pscale, pzero)
from explicit import op_H, op_E1, op_E2, op_P  # noqa: E402
from canon import op_D, op_V1, op_V2, op_M, defect, keyset, tovec  # noqa: E402

RAW = enumerate_terms(2)


def raw_op(S, k):
    return lambda A, B: eval_term_sym(S, RAW[k], A, B)


def combo(S, parts):
    def op(A, B):
        acc = [[pzero() for _ in range(S.n)] for _ in range(S.n)]
        for c, f in parts:
            T = f(A, B)
            for i in range(S.n):
                for j in range(i, S.n):
                    acc[i][j] = padd(acc[i][j], pscale(T[i][j], c))
                    acc[j][i] = acc[i][j]
        return acc
    return op


def curated(S):
    R = lambda k: raw_op(S, k)          # noqa: E731
    N = lambda f: (lambda A, B: f(S, A, B))  # noqa: E731
    L = [
        ("H", N(op_H)),
        ("E1+E2", combo(S, [(1, N(op_E1)), (1, N(op_E2))])),
        ("P", N(op_P)),
        # Delta_A A' + Delta_{A'} A
        ("G1", combo(S, [(1, R(43)), (1, R(118))])),
        # d_{zeta'} A + d_{zeta} A'
        ("G2", combo(S, [(1, R(61)), (-1, R(87))])),
        ("G3", combo(S, [(1, R(55)), (1, R(64))])),
        ("G4", R(68)),
        ("G5", combo(S, [(1, N(op_V1)), (1, N(op_V2))])),
        ("M", N(op_M)),
        ("D", N(op_D)),
    ]
    L += [(f"t{k}", R(k)) for k in range(len(RAW))]
    return L


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    nt = nt or None
    S = Symp(m)
    rng = random.Random(1618 + m)
    ops = curated(S)
    print(f"m = {m}: {len(ops)} candidate operations")

    probes = [(random_sym_field(S, 3, rng, nterms=nt), random_sym_field(S, 3, rng, nterms=nt))
              for _ in range(3)]
    allvals = [[o(A, B) for (A, B) in probes] for _, o in ops]
    ks = keyset(S, allvals)
    vecs = [tovec(S, v, ks) for v in allvals]
    chosen, basis = [], []
    for k, v in enumerate(vecs):
        if rank(basis + [v]) > len(basis):
            basis.append(v)
            chosen.append(k)
    print(f"independent: {len(chosen)}; curated ones kept: "
          + ", ".join(ops[k][0] for k in chosen if k < 10))

    rows = []
    cub = monomial_polys(S, 3)[:1]
    for s in range(nsamp):
        A = random_sym_field(S, 3, rng, nterms=nt)
        B = random_sym_field(S, 3, rng, nterms=nt)
        for f in cub:
            ds = [defect(S, ops[k][1], f, A, B) for k in chosen]
            keys = set()
            for D in ds:
                for i in range(S.n):
                    for j in range(i, S.n):
                        for e in D[i][j]:
                            keys.add((i, j, e))
            for (i, j, e) in sorted(keys):
                rows.append([D[i][j].get(e, Fraction(0)) for D in ds])
    ns = nullspace(rows)
    print(f"equations {len(rows)};  invariant operators of order 2: {len(ns)}")
    for v in ns:
        piv = next(c for c in v if c != 0)
        v = [Fraction(c) / piv for c in v]
        nz = [(ops[k][0], c) for k, c in zip(chosen, v) if c]
        print(f"   Gz  ({len(nz)} nonzero terms):")
        for nm, c in nz:
            print(f"      {str(c):>8} * {nm}")


if __name__ == "__main__":
    main()
