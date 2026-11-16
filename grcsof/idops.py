#!/usr/bin/env python3
"""identification of the as-invariant operators found by kasV.py:

   grad : F --> T(Lam^1),   (grad phi)_i = sum_j C_{ij} D_j(phi)   (C = B^{-1})
   div  : T(Lam^1) --> F,   div(phi)     = sum_i D_i(phi_i)
   Dirac: T(S^+) --> T(S^-),the Clifford contraction of the covariant derivative

and the check that grad.div is the second basis vector of the order-2 space
Hom(T(Lam^1), T(Lam^1))."""
from fractions import Fraction as Fr
from kasV import TYPES as T, usolve, uterms, ulabel, words
from kas6 import ALG, B6

L1 = T['L1']


def as_matrix(v, ts, V1, V2, N):
    """the operator as a dict (word, a, b) -> coeff."""
    return {t: c for t, c in zip(ts, v) if c}


def compose(P, Q):
    """(word,a,b) dicts: first Q : V1 -> V2, then P : V2 -> V3.
    Words compose by concatenation; here every word is (k, A) and we only ever
    compose words with disjoint A, so the product is (kP+kQ, A_P + A_Q) with
    the Koszul sign of re-sorting."""
    out = {}
    for (wP, b, c), cP in P.items():
        for (wQ, a, b2), cQ in Q.items():
            if b2 != b:
                continue
            kP, AP = wP
            kQ, AQ = wQ
            if set(AP) & set(AQ):
                continue                     # D_i D_i = B_ii d_t = 0 here
            merged = list(AP) + list(AQ)
            sgn = 1
            for x in range(len(merged)):
                for y in range(x + 1, len(merged)):
                    if merged[x] > merged[y]:
                        sgn = -sgn
            w = (kP + kQ, tuple(sorted(merged)))
            out[(w, a, c)] = out.get((w, a, c), 0) + sgn * cP * cQ
    return {k: v for k, v in out.items() if v}


if __name__ == '__main__':
    C = ALG.C
    ts1, ns1 = usolve('as', T['F'], L1, 1, quiet=True)
    ts2, ns2 = usolve('as', L1, T['F'], 1, quiet=True)
    grad = as_matrix(ns1[0], ts1, None, None, 1)
    div = as_matrix(ns2[0], ts2, None, None, 1)
    print("grad =", "  ".join(f"{ulabel(t)}:{c}" for t, c in grad.items()))
    print("  i.e. (grad phi)_i = sum_j C_ij D_j(phi) ? ",
          all(grad.get((((0,), (j,))[0] and (0, (j,)), 0, i), 0) == C[i][j]
              for i in range(6) for j in range(6)
              if C[i][j] or ((0, (j,)), 0, i) in grad))
    print("div  =", "  ".join(f"{ulabel(t)}:{c}" for t, c in div.items()))
    comp = compose(grad, div)
    print("\ngrad o div =", "  ".join(f"{ulabel(t)}:{c}"
                                      for t, c in sorted(comp.items(), key=str)))
    ts, ns = usolve('as', L1, L1, 2, quiet=True)
    print(f"\nHom(T(Lam1),T(Lam1)) in order 2: dim {len(ns)}")
    idx = {t: i for i, t in enumerate(ts)}
    vc = [0] * len(ts)
    ok = True
    for t, c in comp.items():
        if t not in idx:
            ok = False
            print("   composite has a term outside the ansatz:", t)
        else:
            vc[idx[t]] = c
    # is vc in the span of ns?
    rows = [list(map(Fr, v)) for v in ns]
    tgt = list(map(Fr, vc))
    piv = []
    for r in rows:
        p = next(i for i, x in enumerate(r) if x)
        piv.append(p)
    w = tgt[:]
    for r, p in zip(rows, piv):
        if w[p]:
            c = w[p] / r[p]
            w = [a - c * b for a, b in zip(w, r)]
    print("grad o div lies in the order-2 space:", ok and not any(w))
    # and it is not a multiple of d_t
    print("grad o div is not a multiple of d_t . id:",
          any(len(t[0][1]) == 2 for t in comp))


# ---- numerical identification: evaluate the operators, do not compose words

def opfun(v, ts, V1, V2):
    """the coefficient vector as an actual map (list of superfunctions of
    length V1.dim) -> (list of length V2.dim)."""
    from superN import add, smul, mono
    W = {}
    for (w, a, b), c in zip(ts, v):
        if c:
            W.setdefault(w, ALG.Dword(w[1], w[0]))

    def P(phi):
        out = [{} for _ in range(V2.dim)]
        for (w, a, b), c in zip(ts, v):
            if c and phi[a]:
                out[b] = add(out[b], smul(c, W[w](phi[a])))
        return out
    return P


def identify():
    from superN import add, smul, mono
    ts1, ns1 = usolve('as', T['F'], L1, 1, quiet=True)
    ts2, ns2 = usolve('as', L1, T['F'], 1, quiet=True)
    ts, ns = usolve('as', L1, L1, 2, quiet=True)
    grad = opfun(ns1[0], ts1, T['F'], L1)
    div = opfun(ns2[0], ts2, L1, T['F'])
    basis = [opfun(v, ts, L1, L1) for v in ns]
    # sample: apply to phi = monomial in slot a
    mons = ALG.monos(2)
    rows, rhs = [], []
    for a in range(L1.dim):
        for m in mons:
            phi = [{} for _ in range(L1.dim)]
            phi[a] = mono(*m)
            gd = grad(div(phi))
            bs = [Pi(phi) for Pi in basis]
            keys = set()
            for x in [gd] + bs:
                for b in range(L1.dim):
                    keys |= {(b, k) for k in x[b]}
            for (b, k) in sorted(keys, key=str):
                rows.append([x[b].get(k, 0) for x in bs])
                rhs.append(gd[b].get(k, 0))
    # least squares over Q: solve rows . c = rhs exactly
    n = len(ns)
    M = [r[:] + [y] for r, y in zip(rows, rhs)]
    piv = []
    r0 = 0
    for c in range(n):
        p = next((i for i in range(r0, len(M)) if M[i][c]), None)
        if p is None:
            continue
        M[r0], M[p] = M[p], M[r0]
        pv = Fr(M[r0][c])
        M[r0] = [Fr(x) / pv for x in M[r0]]
        for i in range(len(M)):
            if i != r0 and M[i][c]:
                f = Fr(M[i][c])
                M[i] = [Fr(x) - f * Fr(y) for x, y in zip(M[i], M[r0])]
        piv.append(c)
        r0 += 1
    sol = [0] * n
    for i, c in enumerate(piv):
        sol[c] = M[i][n]
    ok = all(not any(M[i][:n]) or True for i in range(len(M)))
    res = [Fr(M[i][n]) for i in range(r0, len(M))]
    print("grad o div =", " + ".join(f"({sol[i]})*e{i}" for i in range(n)),
          " -- exact:", not any(res))
    return sol


if __name__ == '__main__':
    print()
    identify()
