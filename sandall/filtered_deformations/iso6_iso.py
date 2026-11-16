#!/usr/bin/env python3
"""Explicit isomorphism k^(1)(3;1) -> (deform along the (-6;0) cocycle), if any:
  1) in the deform, a split torus T' carrying a codim-3 subalgebra L0 of depth 2;
  2) its Weisfeiler filtration L_{-2} > L_{-1} > L_0 > L_1 > ...;
  3) a graded isomorphism psi: k^(1) -> gr (psi_{-1} runs over GL(2;F_3), the
     rest is forced by transitivity);
  4) the filtered deformation is killed step by step: mu = mu_0 + mu_1 + ...,
     mu_s = delta f_s solved by linear algebra, phi = prod (1 + f_s);
  5) phi is checked against the ORIGINAL structure constants on all pairs."""
import sys, time
import numpy as np
from itertools import product
import deform as D
from iso6 import T, LABEL, rank3, ad
from iso6_tori import nullsp, inv3
from iso6_wf import random_tori, search, rebase
from k3 import rref
P = 3


def span_rows(rows, n):
    rows = [list(map(int, r)) for r in rows]
    return [np.array(r, dtype=np.int64) for r in rref(rows, n)[0]] if rows else []


def contains(L, v):
    if not L:
        return not np.any(v % P)
    return rank3(np.array(L + [v])) == len(L)


def brk(tab, x, y):
    return np.einsum("i,j,ijk->k", x, y, tab) % P


def intersect_pre(tab, Lcur, Lm1, n):
    """{x in Lcur : [x, Lm1] subset Lcur}."""
    Lc = np.array(Lcur)
    # complement coordinates: project mod Lcur via a basis completion
    comp = [np.eye(n, dtype=np.int64)[i] for i in range(n)]
    full = list(Lcur)
    compl = []
    for e in comp:
        if not contains(full, e):
            full.append(e); compl.append(e)
    Binv = inv3(np.array(full).T)          # coords in basis full
    k = len(Lcur)
    rows = []
    for y in Lm1:
        M = np.array([Binv @ brk(tab, x, y) % P for x in Lcur])   # k x n
        rows.append(M[:, k:].T)                                   # (n-k) x k
    A = np.concatenate(rows) % P
    ns = nullsp(A)
    return span_rows([sum(int(c) * x for c, x in zip(s, Lcur)) % P for s in ns], n)


def main():
    nm = sys.argv[1] if len(sys.argv) > 1 else "E6_1"
    C = T[nm]
    n = 26
    print("=" * 72); print("target:", LABEL[nm])
    # 1) torus with a depth-2 codim-3 subalgebra
    found = None
    for seed in range(5, 40):
        for t1, t2 in random_tori(C, 12, seed):
            res, (alg, ws, B) = search(C, t1, t2)
            for c, L0 in ws:
                dims = D.filtration(alg, L0)
                if len(dims) == 3:
                    found = (alg, L0, B); break
            if found: break
        if found: break
    alg, L0, B = found
    tab = np.zeros((n, n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k, v in alg.c[i][j].items():
                tab[i, j, k] = v
    L0 = span_rows(L0, n)
    # 2) filtration.  L_{-1}: minimal L0-module strictly above L0 (weight vectors)
    best = None
    for w, idxs in D.cls.items():
        for co in product(range(P), repeat=len(idxs)):
            if not any(co): continue
            v = [0] * n
            for i, c in zip(idxs, co): v[i] = c
            if D.contains([list(map(int, r)) for r in L0], v): continue
            sp = D.module_gen(alg, [list(map(int, r)) for r in L0], v)
            if best is None or len(sp) < len(best): best = sp
    Lm1 = span_rows(best, n)
    layers = {-2: span_rows(list(np.eye(n, dtype=np.int64)), n), -1: Lm1, 0: L0}
    d = 0
    while layers[d]:
        layers[d + 1] = intersect_pre(tab, layers[d], Lm1, n)
        d += 1
    top = d - 1
    dims = {d: len(layers[d]) - len(layers[d + 1]) for d in range(-2, top + 1)}
    print("   filtration dims gr_d:", dims)
    # adapted basis
    E, deg = [], []
    for d in range(-2, top + 1):
        cur = list(layers[d + 1])
        for r in layers[d]:
            if not contains(cur, r):
                cur.append(r); E.append(r); deg.append(d)
    E = np.array(E) % P
    mu = rebase(tab, E)                     # structure constants in basis E
    deg = np.array(deg)
    # filtration check and gr
    gr = np.zeros_like(mu)
    for a in range(n):
        for b in range(n):
            for k in range(n):
                if mu[a, b, k]:
                    assert deg[k] >= deg[a] + deg[b], "not a filtration"
                    if deg[k] == deg[a] + deg[b]:
                        gr[a, b, k] = mu[a, b, k]
    # k^(1) with its degrees
    K = T["k1"]
    kdeg = np.array([w[0] for w in D.g.wt])
    kdims = {d: int((kdeg == d).sum()) for d in range(-2, kdeg.max() + 1)}
    print("   k^(1) dims g_d:     ", kdims)
    assert kdims == dims
    # 3) graded isomorphism psi: k -> gr  (matrix Psi, column = image of k-basis vector)
    kidx = {d: [i for i in range(n) if kdeg[i] == d] for d in kdims}
    gidx = {d: [i for i in range(n) if deg[i] == d] for d in dims}
    psi_ok = None
    for m in product(range(P), repeat=4):
        M2 = np.array(m).reshape(2, 2)
        if rank3(M2) < 2: continue
        Psi = np.zeros((n, n), dtype=np.int64)
        for c, i in enumerate(kidx[-1]):
            for r, j in enumerate(gidx[-1]):
                Psi[j, i] = M2[r, c]
        # degree -2
        u1, u2 = kidx[-1]
        z = K[u1, u2]
        (i2,) = kidx[-2]
        img = brk(gr, Psi[:, u1], Psi[:, u2])
        if not img.any(): continue
        Psi[:, i2] = img * pow(int(z[i2]), 1, P) % P    # z[i2] in {1,2}, inverse = itself
        ok = True
        for dd in range(0, top + 1):
            for i in kidx[dd]:
                # solve y in gr_dd : [y, Psi u] = Psi [e_i, u] for u in g_-1, g_-2
                cols = gidx[dd]
                A, bvec = [], []
                for u in kidx[-1] + kidx[-2]:
                    lhs = np.array([brk(gr, np.eye(n, dtype=np.int64)[c], Psi[:, u]) for c in cols]).T
                    rhs = Psi @ K[i, u] % P
                    A.append(lhs); bvec.append(rhs)
                A = np.concatenate(A) % P; bvec = np.concatenate(bvec) % P
                aug = np.concatenate([A, bvec[:, None]], axis=1)
                if rank3(aug) != rank3(A): ok = False; break
                assert rank3(A) == len(cols)
                ns = nullsp(aug)
                sol = [s for s in ns if s[-1] == P - 1] or [s for s in ns if s[-1]]
                s = sol[0]
                s = s * (P - 1 if s[-1] == 1 else 1) % P   # normalise last coord to -1
                y = np.zeros(n, dtype=np.int64)
                for c, v in zip(cols, s[:-1]): y[c] = v
                Psi[:, i] = y
            if not ok: break
        if not ok: continue
        # full check that Psi is a graded homomorphism k -> gr
        lhs = np.einsum("ia,jb,ijk->abk", Psi, Psi, gr) % P   # [Psi e_a, Psi e_b] in gr
        rhs = np.einsum("kc,abc->abk", Psi, K) % P
        if np.array_equal(lhs, rhs) and rank3(Psi) == n:
            psi_ok = Psi; break
    print("   graded isomorphism k^(1) -> gr found:", psi_ok is not None)
    if psi_ok is None:
        return
    Psi = psi_ok
    # transport: basis F = columns of Psi (in basis E); mu in basis F, with degrees kdeg
    Fi = inv3(Psi)
    muF = np.einsum("ia,jb,ijk->abk", Psi, Psi, mu) % P
    muF = np.einsum("ck,abk->abc", Fi, muF) % P
    # 4) kill the higher terms: mu_s = delta f_s
    Phi = np.eye(n, dtype=np.int64)          # k-basis -> F-basis
    cur = muF.copy()
    for s in range(1, 3 * (top + 2) + 1):
        ms = np.zeros_like(cur)
        for a in range(n):
            for b in range(n):
                for k in range(n):
                    if cur[a, b, k] and kdeg[k] - kdeg[a] - kdeg[b] == s:
                        ms[a, b, k] = cur[a, b, k]
                    if cur[a, b, k] and kdeg[k] - kdeg[a] - kdeg[b] < s and kdeg[k] - kdeg[a] - kdeg[b] > 0:
                        raise AssertionError("lower term survived")
        if not ms.any():
            continue
        # unknown f: f[k, a] with kdeg[k] - kdeg[a] == s
        unk = [(k, a) for a in range(n) for k in range(n) if kdeg[k] - kdeg[a] == s]
        pos = {u: t for t, u in enumerate(unk)}
        rows, rhs = [], []
        for a in range(n):
            for b in range(a + 1, n):
                for k in range(n):
                    if kdeg[k] - kdeg[a] - kdeg[b] != s: continue
                    r = np.zeros(len(unk), dtype=np.int64)
                    # (delta f)(a,b)_k = [f a, b]_k + [a, f b]_k - f([a,b])_k
                    for c in range(n):
                        if (c, a) in pos and K[c, b, k]: r[pos[(c, a)]] += K[c, b, k]
                        if (c, b) in pos and K[a, c, k]: r[pos[(c, b)]] += K[a, c, k]
                        if K[a, b, c] and (k, c) in pos: r[pos[(k, c)]] -= K[a, b, c]
                    rows.append(r % P); rhs.append(ms[a, b, k])
        A = np.array(rows) % P; bb = np.array(rhs) % P
        aug = np.concatenate([A, bb[:, None]], axis=1)
        solvable = rank3(aug) == rank3(A)
        print(f"   shift {s}: {int((ms != 0).sum())} terms, {len(unk)} unknowns, coboundary: {solvable}")
        if not solvable:
            print("   OBSTRUCTED -> not a trivial filtered deformation"); return
        ns = nullsp(aug)
        sol = next(v for v in ns if v[-1])
        sol = sol * (P - 1 if sol[-1] == 1 else 1) % P
        f = np.zeros((n, n), dtype=np.int64)
        for (k, a), v in zip(unk, sol[:-1]): f[k, a] = v
        # new basis: x -> x - f x  (so that the degree-s part is removed)
        G = (np.eye(n, dtype=np.int64) - f) % P
        Gi = inv3(G)
        cur = np.einsum("ia,jb,ijk->abk", G, G, cur) % P
        cur = np.einsum("ck,abk->abc", Gi, cur) % P
        Phi = Phi @ G % P
    print("   remaining structure == k^(1):", np.array_equal(cur, K % P))
    # 5) the isomorphism in the ORIGINAL basis of the deform:
    #    k-basis vector a -> sum over F -> E -> rebased -> original
    Iso = (B.T @ (E.T @ (Psi @ Phi % P) % P) % P) % P    # columns = images in original coords
    lhs = np.einsum("ia,jb,ijk->abk", Iso, Iso, C) % P
    rhs = np.einsum("kc,abc->abk", Iso, K) % P
    print(f"   explicit phi: k^(1)(3;1) -> {nm}: bijective {rank3(Iso) == n}, "
          f"homomorphism on all {n*n} pairs: {np.array_equal(lhs, rhs)}")
    np.save(f"iso6_phi_{nm}.npy", Iso)


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"   ({time.time()-t0:.0f}s)")
