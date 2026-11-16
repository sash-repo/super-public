#!/usr/bin/env python3
"""Chevalley-Eilenberg cohomology H^*(h; g) over F_3, split by the bigrading
(degree; pq-weight).  h is a graded subalgebra of g, the module is g itself.

Cochains: C^n = Lambda^n(h)^* (x) g, basis  (S, m)  with S a sorted n-tuple of
h-basis indices and m a g-basis index; its weight is  wt(m) - sum_{i in S} wt(i).
The differential preserves the weight, so every computation splits into small
blocks, one per weight.
"""
from itertools import combinations
from k3 import P, rref, rank, nullspace


def sign_sort(idx):
    idx = list(idx)
    s = 1
    for i in range(len(idx)):
        for j in range(len(idx) - 1 - i):
            if idx[j] > idx[j + 1]:
                idx[j], idx[j + 1] = idx[j + 1], idx[j]
                s = -s
            elif idx[j] == idx[j + 1]:
                return None, 0
    return tuple(idx), s


class Complex:
    def __init__(self, g, hbasis):
        self.g = g
        self.h = list(hbasis)
        self.pos = {v: i for i, v in enumerate(self.h)}
        self.hwt = [g.wt[v] for v in self.h]

    def cochains(self, n):
        return [(S, m) for S in combinations(range(len(self.h)), n)
                for m in range(self.g.n)]

    def weight(self, S, m):
        w = list(self.g.wt[m])
        for i in S:
            w = [a - b for a, b in zip(w, self.hwt[i])]
        return tuple(w)

    def diff(self, n):
        """columns of d: C^n -> C^{n+1}; dict source -> {target: coeff}."""
        g, h = self.g, self.h
        cols = {}

        def add(src, tgt, co):
            co %= P
            if co:
                d = cols.setdefault(src, {})
                d[tgt] = (d.get(tgt, 0) + co) % P
                if d[tgt] == 0:
                    del d[tgt]

        for T in combinations(range(len(h)), n + 1):
            for i in range(n + 1):
                rest = T[:i] + T[i + 1:]
                sgn = (-1) ** i
                xi = h[T[i]]
                for m in range(g.n):
                    for k, c in g.c[xi][m].items():
                        add((rest, m), (T, k), sgn * c)
            for i in range(n + 1):
                for j in range(i + 1, n + 1):
                    rest = tuple(t for s, t in enumerate(T) if s not in (i, j))
                    sgn = (-1) ** (i + j)
                    for k, c in g.c[h[T[i]]][h[T[j]]].items():
                        if k not in self.pos:
                            raise ValueError("h is not a subalgebra")
                        S, s2 = sign_sort((self.pos[k],) + rest)
                        if s2 == 0:
                            continue
                        for m in range(g.n):
                            add((S, m), (T, m), sgn * s2 * c)
        return cols

    def blocks(self, n):
        """dict weight -> list of C^n basis elements."""
        out = {}
        for S, m in self.cochains(n):
            out.setdefault(self.weight(S, m), []).append((S, m))
        return out

    def cohomology(self, n):
        """dict weight -> (dimH, basis-of-that-weight, cocycle reps, coboundary basis)."""
        Bn = self.blocks(n)
        Bn1 = self.blocks(n - 1) if n else {}
        dn = self.diff(n)
        dn1 = self.diff(n - 1) if n else {}

        out = {}
        for w, src in sorted(Bn.items()):
            pos = {c: i for i, c in enumerate(src)}
            N = len(src)
            # rows of d^n on this block (targets as rows)
            rowmap = {}
            for i, s in enumerate(src):
                for tgt, c in dn.get(s, {}).items():
                    rowmap.setdefault(tgt, [0] * N)[i] = c
            rows = list(rowmap.values())
            Z = nullspace(rows, N) if rows else [[1 if i == j else 0 for i in range(N)]
                                                 for j in range(N)]
            B = []
            for s0 in Bn1.get(w, []):
                r = [0] * N
                for tgt, c in dn1.get(s0, {}).items():
                    if tgt in pos:
                        r[pos[tgt]] = c
                if any(r):
                    B.append(r)
            Bred = rref(B, N)[0] if B else []
            dimH = len(Z) - len(Bred)
            if dimH:
                reps, cur = [], [r[:] for r in Bred]
                rk = rank(cur, N) if cur else 0
                for z in Z:
                    if rank(cur + [z], N) > rk:
                        cur.append(z)
                        rk += 1
                        reps.append(z)
                out[w] = (dimH, src, reps, Bred)
        return out


def to_fun(cx, vec, n):
    """coordinates on cx.cochains(n) -> dict sorted-tuple -> g-vector."""
    phi = {}
    for co, (S, m) in zip(vec, cx.cochains(n)):
        if co:
            phi.setdefault(S, [0] * cx.g.n)
            phi[S][m] = (phi[S][m] + co) % P
    return phi


def from_fun(cx, phi, n):
    src = cx.cochains(n)
    pos = {c: i for i, c in enumerate(src)}
    out = [0] * len(src)
    for S, v in phi.items():
        for m, co in enumerate(v):
            if co:
                out[pos[(S, m)]] = co
    return out


def act(cx, x, vec, n=2):
    """action of x in the normaliser of h on C^n(h; g):
       (x.phi)(a_1..a_n) = [x, phi(a_1..a_n)] - sum_i phi(..,[x,a_i],..).
    (Verified to commute with the differential.)"""
    from itertools import combinations
    g = cx.g
    phi = to_fun(cx, vec, n)
    out = {}
    zero = [0] * g.n
    for T in combinations(range(len(cx.h)), n):
        e = [1 if s == x else 0 for s in range(g.n)]
        val = g.bracket(e, phi.get(T, zero))
        for i in range(n):
            for k, c in g.c[x][cx.h[T[i]]].items():
                if k not in cx.pos:
                    raise ValueError("x does not normalise h")
                S, s2 = sign_sort(T[:i] + (cx.pos[k],) + T[i + 1:])
                if s2 == 0:
                    continue
                v = phi.get(S, zero)
                val = [(a - c * s2 * b) % P for a, b in zip(val, v)]
        if any(val):
            out[T] = val
    return from_fun(cx, out, n)


def invariants(cx, H, g0, n=2):
    """basis of (H^n(h;g))^{g_0}, in coordinates on cx.cochains(n)."""
    src = cx.cochains(n)
    pos = {c: i for i, c in enumerate(src)}
    N = len(src)

    def embed(vec, s):
        out = [0] * N
        for co, c in zip(vec, s):
            out[pos[c]] = co
        return out

    Z = [embed(r, s) for w, (d, s, reps, B) in sorted(H.items()) for r in reps]
    dprev = cx.diff(n - 1)
    Ball = []
    for s0, col in dprev.items():
        r = [0] * N
        for tgt, c in col.items():
            r[pos[tgt]] = c
        if any(r):
            Ball.append(r)
    Bred = rref(Ball, N)[0] if Ball else []
    piv = [next(i for i, x in enumerate(r) if x) for r in Bred]

    def modB(v):
        v = v[:]
        for i, pc in enumerate(piv):
            if v[pc]:
                f = v[pc]
                v = [(a - f * b) % P for a, b in zip(v, Bred[i])]
        return v

    M = []
    for x in g0:
        imgs = [modB(act(cx, x, z, n)) for z in Z]
        for coord in range(N):
            row = [im[coord] for im in imgs]
            if any(row):
                M.append(row)
    sol = nullspace(M, len(Z)) if M else [[1 if i == j else 0 for i in range(len(Z))]
                                          for j in range(len(Z))]
    out = []
    for s in sol:
        v = [0] * N
        for co, z in zip(s, Z):
            if co:
                v = [(a + co * b) % P for a, b in zip(v, z)]
        out.append(v)
    return out, Z, Bred


def show(vec, src, g, sep=" + "):
    terms = []
    for co, (S, m) in zip(vec, src):
        if co:
            hats = "^".join("\\widehat{%s}" % g.names[i] for i in S)
            c = "" if co == 1 else f"{co}"
            terms.append(f"{c}{g.names[m]}(x){hats}")
    return sep.join(terms) if terms else "0"
