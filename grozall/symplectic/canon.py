"""Read off Grozman's Gz for m > 1 in a canonical, human-readable basis.

Strategy.

*  The 135 contraction schemes of `symp.enumerate_terms(2)` span ALL
   translation-invariant, Sp(2m)-covariant bilinear operators of total order 2
   on covariant symmetric 2-tensor fields.  They are not independent.
*  We put in front of them the natural named operations (H, E1, E2, P, ...),
   greedily pick a maximal independent family in that order, and express the
   invariant operator in it.  The named ones therefore get used first, and the
   answer comes out as close to Grozman's m = 1 shape as it can be.

Usage:  python3 canon.py [m] [nsamples] [nterms-per-entry]
"""

import sys, os, random
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank, solve  # noqa: E402

from symp import (Symp, enumerate_terms, eval_term_sym, random_sym_field,  # noqa: E402
                  monomial_polys, padd, pscale, pzero, pmul, pdiff, _sum_polys)
from explicit import op_H, op_E1, op_E2, op_P, g_fun, dd_fun  # noqa: E402


# ------------------------------------------------------- extra named pieces


def op_D(S, A, B):
    """sym_ij  div(A)_i div(A')_j ,  div(A)_i = sum omega^{pl} d_p A_{il}."""
    n = S.n
    dA = [_sum_polys([pscale(pdiff(A[i][l], p), w) for p, l, w in S.Omnz]) for i in range(n)]
    dB = [_sum_polys([pscale(pdiff(B[i][l], p), w) for p, l, w in S.Omnz]) for i in range(n)]
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = pscale(padd(pmul(dA[i], dB[j]), pmul(dA[j], dB[i])), Fraction(1, 2))
            out[i][j] = s
            out[j][i] = s
    return out


def op_V1(S, A, B):
    """sym_ij sum omega^{pk} omega^{ls} d_p d_i A_{kl} A'_{js}."""
    return _V(S, A, B)


def op_V2(S, A, B):
    return _V(S, B, A)


def _V(S, A, B):
    n = S.n
    raw = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            terms = []
            for p, k, w1 in S.Omnz:
                for l, s_, w2 in S.Omnz:
                    terms.append(pscale(pmul(pdiff(pdiff(A[k][l], p), i), B[j][s_]), w1 * w2))
            raw[i][j] = _sum_polys(terms)
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = pscale(padd(raw[i][j], raw[j][i]), Fraction(1, 2))
            out[i][j] = s
            out[j][i] = s
    return out


def op_M(S, A, B):
    """sym_ij sum omega^{kr} omega^{ls} d_i A_{kl} d_j A'_{rs}."""
    n = S.n
    raw = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            terms = []
            for k, r, w1 in S.Omnz:
                for l, s_, w2 in S.Omnz:
                    terms.append(pscale(pmul(pdiff(A[k][l], i), pdiff(B[r][s_], j)), w1 * w2))
            raw[i][j] = _sum_polys(terms)
    out = [[pzero() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = pscale(padd(raw[i][j], raw[j][i]), Fraction(1, 2))
            out[i][j] = s
            out[j][i] = s
    return out


def op_K(S, A, B):
    """sym_ij sum omega^{pl} omega^{qs} d_p A_{il} d_q A'_{js}
    -- the 'divergence x divergence with free indices on the tensors'."""
    return op_D(S, A, B)


NAMEDOPS = [("H", op_H), ("E1", op_E1), ("E2", op_E2), ("P", op_P),
            ("D", op_D), ("V1", op_V1), ("V2", op_V2), ("M", op_M)]


# ------------------------------------------------------------------ machinery


def fingerprint(S, opvals, probes):
    """Flatten the values of an operator on the probe pairs into one vector."""
    vec = []
    for T in opvals:
        for i in range(S.n):
            for j in range(i, S.n):
                vec.append(T[i][j])
    return vec


def keyset(S, allvals):
    ks = []
    nprobe = len(allvals[0])
    for pi in range(nprobe):
        s = set()
        for vals in allvals:
            T = vals[pi]
            for i in range(S.n):
                for j in range(i, S.n):
                    for e in T[i][j]:
                        s.add((i, j, e))
        ks.append(sorted(s))
    return ks


def tovec(S, vals, ks):
    row = []
    for pi, kk in enumerate(ks):
        T = vals[pi]
        for (i, j, e) in kk:
            row.append(T[i][j].get(e, Fraction(0)))
    return row


def defect(S, opfun, f, A, B):
    T = opfun(A, B)
    lhs = S.lie_sym2(f, T)
    r1 = opfun(S.lie_sym2(f, A), B)
    r2 = opfun(A, S.lie_sym2(f, B))
    n = S.n
    out = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = padd(lhs[i][j], pscale(padd(r1[i][j], r2[i][j]), -1))
            out[i][j] = s
            out[j][i] = s
    return out


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    nt = nt or None
    S = Symp(m)
    rng = random.Random(2718 + m)

    raw = enumerate_terms(2)
    ops = [(nm, (lambda o: (lambda A, B: o(S, A, B)))(o)) for nm, o in NAMEDOPS]
    ops += [(f"t{k}:{t.label()}", (lambda tt: (lambda A, B: eval_term_sym(S, tt, A, B)))(t))
            for k, t in enumerate(raw)]
    print(f"m = {m}: candidate operations {len(ops)} "
          f"({len(NAMEDOPS)} named + {len(raw)} raw contractions)")

    # ---- greedy independent subset, named ops first
    probes = [(random_sym_field(S, 3, rng, nterms=nt), random_sym_field(S, 3, rng, nterms=nt))
              for _ in range(2)]
    allvals = [[o(A, B) for (A, B) in probes] for _, o in ops]
    ks = keyset(S, allvals)
    vecs = [tovec(S, v, ks) for v in allvals]
    chosen, basis = [], []
    for k, v in enumerate(vecs):
        if rank(basis + [v]) > len(basis):
            basis.append(v)
            chosen.append(k)
    print(f"independent operations: {len(chosen)}")
    print("  named ones kept: " + ", ".join(ops[k][0] for k in chosen if k < len(NAMEDOPS)))

    # ---- invariance system on the chosen basis
    rows = []
    cubics = monomial_polys(S, 3)
    cub = cubics[:1]
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
        print(f"  sample {s+1}: {len(rows)} equations")
    ns = nullspace(rows)
    print(f"invariant operators of order 2 (T(2,0..0)^2 -> T(2,0..0)): {len(ns)}")
    for v in ns:
        # normalise: make the first nonzero coefficient 1
        piv = next(c for c in v if c != 0)
        v = [Fraction(c) / piv for c in v]
        print("   GZ =")
        for k, c in zip(chosen, v):
            if c:
                print(f"      {str(c):>10} * {ops[k][0]}")

    # ---- verify against higher-degree Hamiltonians
    if ns:
        v = ns[0]

        def GZ(A, B):
            acc = [[pzero() for _ in range(S.n)] for _ in range(S.n)]
            for c, k in zip(v, chosen):
                if c == 0:
                    continue
                T = ops[k][1](A, B)
                for i in range(S.n):
                    for j in range(i, S.n):
                        acc[i][j] = padd(acc[i][j], pscale(T[i][j], c))
                        acc[j][i] = acc[i][j]
            return acc

        A = random_sym_field(S, 4, rng, nterms=nt)
        B = random_sym_field(S, 4, rng, nterms=nt)
        for deg in (3, 4):
            ok = all(not any(D[i][j] for i in range(S.n) for j in range(i, S.n))
                     for D in (defect(S, GZ, f, A, B) for f in monomial_polys(S, deg)))
            print(f"  invariance under ALL Hamiltonians of degree {deg}: {'OK' if ok else 'FAILS'}")


if __name__ == "__main__":
    main()
