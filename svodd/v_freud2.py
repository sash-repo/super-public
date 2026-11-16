"""Theorem 4.5.2 (corank 1|6, type V(n)) re-verified with exact FLINT-backed
linear algebra: the top graded piece of the layer, the dimension of the image
of rho(s_f) on it, its invariant bilinear forms, and a generic orbit dimension.
"""
import sys, random
from fractions import Fraction as F
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, nullspace, backend
import stab, orb, sv, t3
from probe import TYPES

print("exactlin backend:", backend(), flush=True)

for n in (3, 4):
    p, q, terms = TYPES["V"](n, 1)
    res = stab.analyse(f"V({n})", p, q, terms, 2 * n + 4, verbose=False)
    L = t3.Layer(res)
    print(f"\n=== V({n}) = x^{n} + x(xi_1xi_2 + xi_3xi_4 + xi_5xi_6)", flush=True)
    print(f"    sdim O_f = ({res['sdim'][0]}|{res['sdim'][1]}), dim Nt_f = {L.n}, "
          f"pieces " + ",".join(f"{d}:{len(L.bydeg[d])}" for d in L.degrees) +
          f", dim rho(s_f) = {res['rhodim']}", flush=True)
    d3 = max(L.degrees)
    idx = L.bydeg[d3]
    k = len(idx)
    print(f"    top piece: degree {d3}, dimension {k}", flush=True)
    # a BASIS of the image on that piece (not the raw generator list)
    sub = []
    for M in L.span:
        S = [[M[i][j] for j in idx] for i in idx]
        if any(any(r) for r in S):
            sub.append(S)
    B = t3.basis_of_span(sub)
    print(f"    dim of the image on the piece = {len(B)}   (dim sp(6) = 21)", flush=True)
    # invariant bilinear forms: M^T Q + Q M = 0, unknowns Q_{ab}
    unk = [(a, b) for a in range(k) for b in range(k)]
    pos = {ab: t for t, ab in enumerate(unk)}
    rows = []
    for S in B:
        for a in range(k):
            for b in range(k):
                row = [F(0)] * len(unk)
                for t in range(k):
                    row[pos[(t, b)]] += S[t][a]
                    row[pos[(a, t)]] += S[t][b]
                if any(row):
                    rows.append(row)
    ns = nullspace(rows)
    print(f"    invariant bilinear forms: {len(ns)}", flush=True)
    for v in ns:
        Q = [[v[pos[(a, b)]] for b in range(k)] for a in range(k)]
        sym = all(Q[a][b] == Q[b][a] for a in range(k) for b in range(k))
        skew = all(Q[a][b] == -Q[b][a] for a in range(k) for b in range(k))
        rk = rank(Q)
        print(f"      one form: {'symmetric' if sym else 'anti-symmetric' if skew else 'neither'}"
              f", rank {rk} of {k}  ({'nondegenerate' if rk == k else 'degenerate'})",
              flush=True)
    # generic orbit dimension on the piece
    rnd = random.Random(11)
    best = 0
    for _ in range(40):
        v = [F(rnd.randint(-6, 6)) for _ in range(k)]
        rows2 = [[sum(S[i][j] * v[j] for j in range(k)) for i in range(k)] for S in B]
        best = max(best, rank(rows2))
    print(f"    generic orbit dimension on the piece = {best}  => codimension {k - best}",
          flush=True)
