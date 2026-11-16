#!/usr/bin/env python3
"""H^2(g_-;g) and its
g_0-invariants for the Hamiltonian algebras h(2k;N) over F_p, in the Poisson model
(generating functions mod constants), standard grading deg f = |a| - 2.
  P = the prolong h(2k;N) = O(2k;N)/K,   S = its second derived (the simple one).
Usage: python3 ham3.py p k N1 ... N2k"""
import sys, itertools
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank, rref

p = int(sys.argv[1]); k = int(sys.argv[2]); N = [int(a) for a in sys.argv[3:]]
n = 2 * k; assert len(N) == n
top = [p ** Ni - 1 for Ni in N]

def binom(a, b):
    r = 1
    for i in range(b):
        r = r * (a - i) // (i + 1)
    return r % p

MON = [a for a in itertools.product(*[range(t + 1) for t in top]) if sum(a) > 0]
IDX = {a: i for i, a in enumerate(MON)}

def mul(a, b):
    c = tuple(x + y for x, y in zip(a, b))
    if any(ci > ti for ci, ti in zip(c, top)): return None, 0
    co = 1
    for x, y in zip(a, b): co = co * binom(x + y, x) % p
    return c, co

def der(a, i):
    if a[i] == 0: return None
    b = list(a); b[i] -= 1; return tuple(b)

def br(a, b):           # {x^a, x^b} as dict mono->coef, constants dropped
    if a == 'E' and b == 'E': return {}
    if a == 'E': return {b: deg(b) % p} if deg(b) % p else {}
    if b == 'E': return {a: (-deg(a)) % p} if deg(a) % p else {}
    out = {}
    for i in range(k):
        for (u, v, s) in ((i, k + i, 1), (k + i, i, -1)):
            da, db = der(a, u), der(b, v)
            if da is None or db is None: continue
            c, co = mul(da, db)
            if c is None or co == 0 or sum(c) == 0: continue
            out[c] = (out.get(c, 0) + s * co) % p
    return {m: c for m, c in out.items() if c}

def deg(a): return 0 if a == 'E' else sum(a) - 2

def span_derived(basis_vecs):
    # basis_vecs: list of dict; returns rref basis (list of dict) of span of brackets
    rows = []
    for u in basis_vecs:
        for v in basis_vecs:
            w = {}
            for m1, c1 in u.items():
                for m2, c2 in v.items():
                    for m, c in br(m1, m2).items():
                        w[m] = (w.get(m, 0) + c1 * c2 * c) % p
            w = {m: c for m, c in w.items() if c}
            if w: rows.append([w.get(m, 0) for m in MON])
    R, piv = rref(rows, p)
    return [{MON[j]: int(r[j]) % p for j in range(len(MON)) if int(r[j]) % p} for r in R if any(int(x) % p for x in r)]

def algebra(which):
    P = [{m: 1} for m in MON]
    if which == 'P': return P
    if which == 'C': return P + [{'E': 1}]
    D1 = span_derived(P)
    if which == 'P1': return D1
    return span_derived(D1)

def analyse(name, G):
    # G: list of homogeneous? make a homogeneous basis per degree
    bydeg = {}
    for v in G:
        ds = {deg(m) for m in v}
        assert len(ds) == 1, "non-homogeneous basis element"
        bydeg.setdefault(ds.pop(), []).append(v)
    dims = {d: len(bydeg[d]) for d in sorted(bydeg)}
    gm = bydeg[-1]; g0 = bydeg.get(0, [])
    assert all(len(v) == 1 for v in gm)
    gmon = [next(iter(v)) for v in gm]
    # coordinates within g_d: express dict in basis bydeg[d] (rref'd, pivot monomials)
    def coords(d, w):
        B = bydeg.get(d, [])
        if not w: return [0] * len(B)
        mons = sorted(set(w) | {m for b in B for m in b}, key=str)
        M = [[b.get(m, 0) for b in B] for m in mons]
        rhs = [w.get(m, 0) for m in mons]
        # solve M x = rhs mod p via nullspace of [M | -rhs]
        A = [row + [(-r) % p] for row, r in zip(M, rhs)]
        ns = nullspace(A, p)
        for v in ns:
            if int(v[-1]) % p:
                inv = pow(int(v[-1]), p - 2, p)
                return [int(x) * inv % p for x in v[:-1]]
        raise ValueError("not in g_%d" % d)
    def act(x, w):   # [x, w], x,w dicts
        out = {}
        for m1, c1 in x.items():
            for m2, c2 in w.items():
                for m, c in br(m1, m2).items():
                    out[m] = (out.get(m, 0) + c1 * c2 * c) % p
        return {m: c for m, c in out.items() if c}
    q = len(gmon)
    res = {}
    Dmin = min(dims) ; Dmax = max(dims) + 3
    tot = 0; totinv = 0
    for D in range(Dmin + 1, Dmax + 1):
        def cbasis(qq):
            out = []
            for S in itertools.combinations(range(q), qq):
                for j in range(len(bydeg.get(D - qq, []))):
                    out.append((S, j))
            return out
        C1, C2, C3 = cbasis(1), cbasis(2), cbasis(3)
        if not C2: continue
        i2 = {c: i for i, c in enumerate(C2)}; i3 = {c: i for i, c in enumerate(C3)}
        def vec_in(Cidx, S, d, w):   # add value w (dict in g_d) at argument set S
            return [(Cidx[(S, j)], c) for j, c in enumerate(coords(d, w)) if c]
        # d1: C1 -> C2 ; (dc)(a,b) = a.c(b) - b.c(a)
        d1 = []
        for (S, j) in C1:
            v = [0] * len(C2); a = S[0]; val = bydeg[D - 1][j]
            for b in range(q):
                if b == a: continue
                x = {gmon[b]: 1}; w = act(x, val)
                T = tuple(sorted((a, b))); sgn = 1 if b < a else -1   # (dc)(b',a') with b'<a'
                # (dc)(u,v)=u.c(v)-v.c(u); c nonzero only on a
                # if T=(b,a): (dc)(b,a) = b.c(a) ; if T=(a,b): (dc)(a,b) = -b.c(a)
                for i, c in vec_in(i2, T, D - 2, w): v[i] = (v[i] + sgn * c) % p
            d1.append(v)
        d2 = []
        for (S, j) in C2:
            v = [0] * len(C3); val = bydeg[D - 2][j]
            for b in range(q):
                if b in S: continue
                T = tuple(sorted(S + (b,)))
                pos = T.index(b); sgn = 1 if pos % 2 == 0 else -1
                w = act({gmon[b]: 1}, val)
                for i, c in vec_in(i3, T, D - 3, w): v[i] = (v[i] + sgn * c) % p
            d2.append(v)
        # Z = ker d2 (rows combos): nullspace of transpose
        if C3:
            Z = nullspace([list(r) for r in zip(*d2)], p) if d2 else []
        else:
            Z = [[1 if i == j else 0 for i in range(len(C2))] for j in range(len(C2))]
        Z = [[int(x) % p for x in z] for z in Z]
        rB = rank(d1, p) if d1 else 0
        h = len(Z) - rB
        if h == 0: continue
        # invariants
        Brows = [r for r in d1]
        RB, pivB = rref(Brows, p) if Brows else ([], [])
        RB = [[int(x) % p for x in r] for r in RB if any(int(x) % p for x in r)]
        pivB = pivB[:len(RB)]
        def reduce(v):
            v = v[:]
            for r, pc in zip(RB, pivB):
                if v[pc]:
                    f = v[pc] * pow(r[pc], p - 2, p) % p
                    v = [(a - f * b) % p for a, b in zip(v, r)]
            return v
        # g_0 action on cochains computed directly: (X.c)(u,v)=[X,c(u,v)]-c([X,u],v)-c(u,[X,v])
        def act2(X, z):
            out = [0] * len(C2)
            full = {}
            for idx, c in enumerate(z):
                if c: full[C2[idx]] = c
            # c as function on pairs (a,b) a<b -> dict value
            cval = {}
            for (S, j), c in full.items():
                d = cval.setdefault(S, {})
                for m, cf in bydeg[D - 2][j].items(): d[m] = (d.get(m, 0) + c * cf) % p
            def cv(a, b):
                if a == b: return {}
                if a < b: return cval.get((a, b), {})
                return {m: (-c) % p for m, c in cval.get((b, a), {}).items()}
            Xg = []
            for a in range(q):
                xa = act(X, {gmon[a]: 1}); Xg.append({gmon.index(m): c for m, c in xa.items()})
            for S in itertools.combinations(range(q), 2):
                a, b = S; w = {}
                for m, c in act(X, cv(a, b)).items(): w[m] = (w.get(m, 0) + c) % p
                for a2, c in Xg[a].items():
                    for m, cc in cv(a2, b).items(): w[m] = (w.get(m, 0) - c * cc) % p
                for b2, c in Xg[b].items():
                    for m, cc in cv(a, b2).items(): w[m] = (w.get(m, 0) - c * cc) % p
                w = {m: c for m, c in w.items() if c}
                for i, c in vec_in(i2, S, D - 2, w): out[i] = (out[i] + c) % p
            return out
        cols = []
        for z in Z:
            col = []
            for X in g0: col += reduce(act2(X, z))
            cols.append(col)
        M = [list(r) for r in zip(*cols)]
        ninv = len(nullspace(M, p)) - rB if M else len(Z) - rB
        res[D] = (h, ninv); tot += h; totinv += ninv
    print(f"{name}: dims {dims}, dim={sum(dims.values())}")
    print(f"   H^2(g_-;g) = {tot} by degree (dim, g0-inv): {res};  invariants {totinv}")

for which in sys.argv[0:0] or ('C', 'P', 'S'):
    analyse(f"p={p} k={k} N={tuple(N)} { {'C':'ch = (V,csp)_*','P':'prolong h','S':'simple h^(2)'}[which] }", algebra(which))
