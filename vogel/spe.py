#!/usr/bin/env python3
"""Invariant bilinear forms on spe(n) (p does not divide n) and pspe(n) (p | n)
over F_p. Unknowns B(e_i,e_j) restricted by torus weights; invariance B([x,y],z)=B(x,[y,z])
imposed for x in the whole basis. Reports the dimension of the space of invariant forms,
split by parity and by Z-degree (deg C=-1, A=0, B=+1), and the max rank of a form."""
import sys, itertools
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace, rank

def build(n, p, evenonly=False):
    N = 2*n
    quot = False  # p | n: diag(1_n,-1_n) lies in spe(n) but is the grading element, not central
    basis = []   # (matrix as dict, parity, degree)
    def E(i, j): return {(i, j): 1}
    def add(a, b, d=None):
        r = dict(a)
        for k, v in b.items(): r[k] = (r.get(k, 0) + v) % p
        return {k: v for k, v in r.items() if v}
    for i in range(n):
        for j in range(n):
            if i != j: basis.append((add(E(i, j), {(n+j, n+i): p-1}, {}), 0, 0))
    nh = n-2 if quot else n-1
    for k in range(nh):
        m = {(k, k): 1, (k+1, k+1): p-1, (n+k, n+k): p-1, (n+k+1, n+k+1): 1}
        basis.append((m, 0, 0))
    for i in range(n):
        for j in range(i, n):
            m = {(i, n+j): 1, (j, n+i): 1} if i != j else {(i, n+i): 2 % p}
            basis.append((m, 1, 1))
    for i in range(n):
        for j in range(i+1, n):
            basis.append(({(n+i, j): 1, (n+j, i): p-1}, 1, -1))
    def mul(a, b):
        r = {}
        for (i, k), v in a.items():
            for (k2, j), w in b.items():
                if k == k2: r[(i, j)] = (r.get((i, j), 0) + v*w) % p
        return r
    def br(x, y):
        (a, pa, _), (b, pb, _) = x, y
        s = -1 if (pa and pb) else 1
        r = mul(a, b)
        for k, v in mul(b, a).items(): r[k] = (r.get(k, 0) - s*v) % p
        return {k: v for k, v in r.items() if v}
    idx = {}
    off = []
    for t, (m, pa, dg) in enumerate(basis):
        pass
    def coords(M):
        c = [0]*len(basis); t = 0
        for i in range(n):
            for j in range(n):
                if i != j: c[t] = M.get((i, j), 0) % p; t += 1
        a = [M.get((i, i), 0) % p for i in range(n)]
        if quot: a = [(x - a[-1]) % p for x in a]
        s = 0
        for k in range(nh):
            s = (s + a[k]) % p; c[t] = s; t += 1
        for i in range(n):
            for j in range(i, n):
                v = M.get((i, n+j), 0) % p
                if i == j: v = v * pow(2, -1, p) % p
                c[t] = v; t += 1
        for i in range(n):
            for j in range(i+1, n):
                c[t] = M.get((n+i, j), 0) % p; t += 1
        return c
    # torus weights: values of diagonal elements h_k on each basis element
    D = len(basis)
    hs = [basis[t] for t in range(D) if t >= n*(n-1) and t < n*(n-1)+nh]
    def wt(x):
        return tuple(tuple(coords(br(h, x))) for h in hs)  # cheap but fine: use eigenvalue
    W = []
    for x in basis:
        w = []
        for h in hs:
            c = coords(br(h, x)); cx = coords(x[0])
            # eigenvalue: ratio on a nonzero coord
            k = next(i for i, v in enumerate(cx) if v)
            w.append(c[k] * pow(cx[k], -1, p) % p)
        W.append(tuple(w))
    R = [i for i in range(D) if not (evenonly and basis[i][1])]
    pairs = [(i, j) for i in R for j in R
             if all((W[i][k] + W[j][k]) % p == 0 for k in range(len(hs)))]
    col = {pr: t for t, pr in enumerate(pairs)}
    S = [[coords(br(basis[i], basis[j])) for j in range(D)] for i in range(D)]
    rows = []
    for x in R:
        for y in R:
            for z in R:
                r = {}
                for k, v in enumerate(S[x][y]):
                    if v and (k, z) in col: r[col[(k, z)]] = (r.get(col[(k, z)], 0) + v) % p
                for k, v in enumerate(S[y][z]):
                    if v and (x, k) in col: r[col[(x, k)]] = (r.get(col[(x, k)], 0) - v) % p
                r = {k: v for k, v in r.items() if v}
                if r: rows.append(r)
        # also pairs not allowed by weight must vanish: automatic (not unknowns)
    # dedupe
    rows = list({tuple(sorted(r.items())): r for r in rows}.values())
    M = [[r.get(c, 0) for c in range(len(pairs))] for r in rows]
    ns = nullspace(M, p) if M else [[1 if c == t else 0 for c in range(len(pairs))] for t in range(len(pairs))]
    par = [b[1] for b in basis]; deg = [b[2] for b in basis]
    out = []
    for v in ns:
        G = [[0]*D for _ in range(D)]
        for (i, j), t in col.items(): G[i][j] = int(v[t]) % p
        kinds = {(par[i]+par[j]) % 2 for (i, j), t in col.items() if int(v[t]) % p}
        degs = {deg[i]+deg[j] for (i, j), t in col.items() if int(v[t]) % p}
        out.append((sorted(kinds), sorted(degs), rank(G, p)))
    return D, sum(1-b[1] for b in basis), sum(b[1] for b in basis), out

for n, p in [(3,5),(4,3),(3,3)]:
    D, e, o, out = build(n, p, evenonly=True)
    print(f"CONTROL sl({n}) inside, p={p}: dim={D} dim inv forms={len(out)} {out}", flush=True)
for n, p in [(3,5),(3,7),(3,3),(4,3),(4,5),(4,7),(5,3),(5,7),(5,5),(6,5),(6,3)]:
    D, e, o, out = build(n, p)
    print(f"n={n} p={p} spe sdim=({e}|{o}) dim inv forms={len(out)} {out}", flush=True)
