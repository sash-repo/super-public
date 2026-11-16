#!/usr/bin/env python3
"""H^2(g_-; g) and (H^2(g_-; g))^{g_0} for the Melikyan algebra (p=5, N=1) in the
two Z-gradings of the text:
   Me : deg t=2, deg y=(1,1,1,1)   (contact, g_- dims 1|4,  g_0 = cvect(1;1))
   me : deg t=3, deg y=(0,1,2,3)   (g_- dims 2|1|2,        g_0 = gl(2))
Bigrading used to split the complex: D = (2;1,1,1,1) and h = (0;-3,-1,1,3)."""
import sys, itertools
from mel5 import *
from mel5 import _rref

mats, forms = setup()
K = Contact(forms[0], (5,) * 5)
comps = prolong(K, [mono(0, 0, 0, 0, 0)], [{lin_index(n): 1} for n in range(4)],
                [mono(1, 0, 0, 0, 0)] + [quad_hamiltonian(K, A) for A in mats])
raw = [f for d in sorted(comps) for f in comps[d]]


def gdeg(g, m):                         # g = (s, c1..c4), field degree
    return g[0] * m[0] + sum(c * b for c, b in zip(g[1:], m[1:])) - g[0]


GD, GH = (2, 1, 1, 1, 1), (0, -3, -1, 1, 3)
# bihomogeneous basis
proj = {}
for f in raw:
    parts = {}
    for m, u in f.items():
        parts.setdefault((gdeg(GD, m), gdeg(GH, m)), {})[m] = u
    for k, g in parts.items():
        proj.setdefault(k, []).append(g)
basis, bideg, comp = [], [], {}
for k in sorted(proj):
    gs = proj[k]
    ms = sorted({m for g in gs for m in g})
    R = _rref([[g.get(m, 0) for m in ms] for g in gs], P)[0]
    for r in R:
        r = [int(x) % P for x in r]
        if any(r):
            comp.setdefault(k, []).append(len(basis))
            basis.append({m: c for m, c in zip(ms, r) if c}); bideg.append(k)
assert len(basis) == 125, len(basis)
N = 125
# coordinate solver per bidegree: pivot structure of rref rows
piv = {}
for k, idx in comp.items():
    ms = sorted({m for i in idx for m in basis[i]})
    piv[k] = [(min(basis[i]), i) for i in idx]    # rref: leading monomial unique
    # check leading monomials distinct and absent from other rows
    for (m0, i) in piv[k]:
        for j in idx:
            if j != i:
                assert basis[j].get(m0, 0) == 0


def coords(f, k):
    f = dict(f); out = {}
    for m0, i in piv.get(k, []):
        c = f.get(m0, 0)
        if c:
            out[i] = c
            for m, u in basis[i].items():
                f[m] = (f.get(m, 0) - c * u) % P
    if any(v % P for v in f.values()):
        raise ValueError("not closed")
    return out


C = [[None] * N for _ in range(N)]
for i in range(N):
    for j in range(N):
        b = K.br(basis[i], basis[j])
        k = (bideg[i][0] + bideg[j][0], bideg[i][1] + bideg[j][1])
        C[i][j] = coords(b, k) if b else {}
print("Me: 125-dim, bracket closed on the bihomogeneous basis", flush=True)


def sign_sort(idx):
    idx = list(idx); s = 1
    for a in range(len(idx)):
        for b in range(len(idx) - 1 - a):
            if idx[b] > idx[b + 1]:
                idx[b], idx[b + 1] = idx[b + 1], idx[b]; s = -s
            elif idx[b] == idx[b + 1]:
                return None, 0
    return tuple(idx), s


def G_of(i, Gr):   # degree of basis element i in grading Gr = a*D + b*h
    a, b = Gr
    return a * bideg[i][0] + b * bideg[i][1]


def analyse(name, Gr, denom):
    deg = lambda i: G_of(i, Gr) // denom
    for i in range(N):
        assert G_of(i, Gr) % denom == 0
    hm = [i for i in range(N) if deg(i) < 0]
    g0 = [i for i in range(N) if deg(i) == 0]
    pos = {v: a for a, v in enumerate(hm)}
    dims = {}
    for i in range(N):
        dims[deg(i)] = dims.get(deg(i), 0) + 1
    print("\n== %s: dims %s" % (name, dict(sorted(dims.items()))), flush=True)

    def chains(n):
        return [(S, m) for S in itertools.combinations(range(len(hm)), n) for m in range(N)]

    def wt(S, m):
        return (bideg[m][0] - sum(bideg[hm[s]][0] for s in S),
                bideg[m][1] - sum(bideg[hm[s]][1] for s in S))

    def diff(n):
        cols = {}
        def add(src, tgt, co):
            co %= P
            if co:
                d = cols.setdefault(src, {}); d[tgt] = (d.get(tgt, 0) + co) % P
        for T in itertools.combinations(range(len(hm)), n + 1):
            for a in range(n + 1):
                rest = T[:a] + T[a + 1:]
                x = hm[T[a]]
                for m in range(N):
                    for k, c in C[x][m].items():
                        add((rest, m), (T, k), (-1) ** a * c)
            for a in range(n + 1):
                for b in range(a + 1, n + 1):
                    rest = tuple(t for s, t in enumerate(T) if s not in (a, b))
                    for k, c in C[hm[T[a]]][hm[T[b]]].items():
                        S, s2 = sign_sort((pos[k],) + rest)
                        if s2:
                            for m in range(N):
                                add((S, m), (T, m), (-1) ** (a + b) * s2 * c)
        return cols

    d1, d2 = diff(1), diff(2)
    blocks2, blocks1 = {}, {}
    for c in chains(2):
        blocks2.setdefault(wt(*c), []).append(c)
    for c in chains(1):
        blocks1.setdefault(wt(*c), []).append(c)
    H = {}
    for w, src in sorted(blocks2.items()):
        ps = {c: a for a, c in enumerate(src)}
        rows = {}
        for a, s in enumerate(src):
            for tgt, c in d2.get(s, {}).items():
                rows.setdefault(tgt, [0] * len(src))[a] = c
        Z = ns(list(rows.values()), len(src))
        B = []
        for s0 in blocks1.get(w, []):
            r = [0] * len(src)
            for tgt, c in d1.get(s0, {}).items():
                r[ps[tgt]] = c
            B.append(r)
        rb = rk(B)
        if len(Z) - rb:
            H[w] = (len(Z) - rb, src, Z, B)
    bydeg = {}
    for w, (d, *_ ) in H.items():
        gd = (Gr[0] * w[0] + Gr[1] * w[1]) // denom
        bydeg[gd] = bydeg.get(gd, 0) + d
    print("dim H^2(g_-;g) =", sum(v[0] for v in H.values()), " by degree:", dict(sorted(bydeg.items())))
    print("  by bidegree (D,h):", {w: v[0] for w, v in sorted(H.items())}, flush=True)
    return hm, g0, H, deg, chains, wt, diff


results = {}
for name, Gr, den in [("Me (contact grading)", (1, 0), 1), ("me (standard grading)", None, None)]:
    if Gr is None:
        # me-degree of monomial = 3a + (0,1,2,3).b - 3 ; express via D=(2;1111), h=(0;-3,-1,1,3):
        # (3;0,1,2,3) = 3/2 D + 1/2 h  ->  2*me = 3D + h
        Gr, den = (3, 1), 2
    results[name] = analyse(name, Gr, den)
import pickle


# ------------------------------------------------------------ checks + invariants
def simple_check():
    """ideal generated by each basis vector of the top degree is everything <=> simple
    (for a transitive graded algebra with irreducible g_{-1}); here: brute ideal closure
    from a single random-ish element is enough to exhibit simplicity sampling."""
    # derived algebra = g ?
    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            if C[i][j]:
                r = [0] * N
                for k, c in C[i][j].items():
                    r[k] = c
                rows.append(r)
    return rk(rows)


print("\ndim [g,g] =", simple_check())


def invariants(name, res, torus_only=False):
    hm, g0, H, deg, chains, wt, diff = res
    if torus_only:
        g0 = [x for x in g0 if bideg[x] == (0, 0)]
    pos = {v: a for a, v in enumerate(hm)}
    d1, d2 = diff(1), diff(2)
    # check d2 d1 = 0
    for s, col in d1.items():
        acc = {}
        for t, c in col.items():
            for u, c2 in d2.get(t, {}).items():
                acc[u] = (acc.get(u, 0) + c * c2) % P
        assert not any(acc.values()), "d^2 != 0"
    B1 = {}
    for s0, col in d1.items():
        for tgt in col:
            pass
    def act(x, vec):   # vec: dict (S,m)->c  -> dict
        out = {}
        for (S, m), c in vec.items():
            for k, cc in C[x][m].items():
                out[(S, k)] = (out.get((S, k), 0) + c * cc) % P
            for a in range(len(S)):
                for k, cc in C[x][hm[S[a]]].items():
                    if k not in pos:
                        raise ValueError("x does not normalise g_-")
                    T, s2 = sign_sort(S[:a] + (pos[k],) + S[a + 1:])
                    if s2:
                        out[(T, m)] = (out.get((T, m), 0) - c * cc * s2) % P
        return {k: v for k, v in out.items() if v}
    # coboundary space of C^2 by block
    Bspace = {}
    for s0, col in d1.items():
        w = None
        v = {t: c for t, c in col.items() if c}
        if v:
            w = wt(*next(iter(v)))
            Bspace.setdefault(w, []).append(v)
    inv_total = 0
    for w, (dimH, src, Z, B) in sorted(H.items()):
        # unknowns: a (len Z) ; condition x.(sum a z) in B for all x in g_0
        zs = [{src[i]: c for i, c in enumerate(z) if c} for z in Z]
        cols_all, rows_meta = [], []
        eqs = []
        for x in g0:
            imgs = [act(x, z) for z in zs]
            wt_t = (w[0] + bideg[x][0], w[1] + bideg[x][1])
            Bs = Bspace.get(wt_t, [])
            keys = sorted({k for im in imgs for k in im} | {k for b in Bs for k in b})
            # a_i * img_i - sum b_j B_j = 0
            for k in keys:
                eqs.append([im.get(k, 0) for im in imgs] + [(-b.get(k, 0)) % P for b in Bs] +
                           [0] * 0)
            # pad later
            rows_meta.append(len(Bs))
        # build a combined system with separate b-variables per x
        nz = len(zs); tot = nz + sum(rows_meta)
        M = []
        off = nz; e = 0
        for xi, x in enumerate(g0):
            nb = rows_meta[xi]
            imgs = [act(x, z) for z in zs]
            wt_t = (w[0] + bideg[x][0], w[1] + bideg[x][1])
            Bs = Bspace.get(wt_t, [])
            keys = sorted({k for im in imgs for k in im} | {k for b in Bs for k in b})
            for k in keys:
                r = [0] * tot
                for i, im in enumerate(imgs):
                    r[i] = im.get(k, 0)
                for j, b in enumerate(Bs):
                    r[off + j] = (-b.get(k, 0)) % P
                M.append(r)
            off += nb
        sol = ns(M, tot)
        # dimension of invariant classes = rank of a-part of solutions modulo coboundaries in block w
        apart = [s[:nz] for s in sol]
        # combos that are themselves coboundaries: add B of block w
        Bw = Bspace.get(w, [])
        vecs = []
        for a in apart:
            v = [0] * len(src)
            for c, z in zip(a, Z):
                v = [(p + c * q) % P for p, q in zip(v, z)]
            vecs.append(v)
        Bm = [[b.get(s, 0) for s in src] for b in Bw]
        dinv = rk(vecs + Bm) - rk(Bm)
        if dinv:
            print("  %s: g_0-invariant classes in bidegree %s: %d" % (name, w, dinv))
            inv_total += dinv
    print("%s: dim (H^2(g_-;g))^{g_0} = %d" % (name, inv_total), flush=True)


for name, res in results.items():
    invariants(name + ' [control: torus of g_0 only]', res, torus_only=True)
    invariants(name, res)
