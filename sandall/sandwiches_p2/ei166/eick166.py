#!/usr/bin/env python3
"""Eick algebra 16:6 = Eick_{16,4} of the CSS library, over F_2.
 1. its maximal subalgebras of the smallest codimension (4) and W-filtrations;
 2. the associated graded algebra gr L = g_{-1}+g_0+g_1, (4|8|4), and whether L ~ gr L;
 3. the g_0-module g_{-1}: g_0 ~ sl(3)?  conjugacy class of its image in gl(4)
    against A. Lebedev's three groups N1, N7, N8 (table irrwithoutLWV) and the
    weighted densities;  is g_1 ~ g_{-1}^* ?
"""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from css import LieF2, rref, in_span, all_elements
from runall import build
from allmin import all_smallest_codim

BASE = os.path.join(os.path.dirname(__file__), "..", "css-bases", "dim16")

def coords(v, basis):
    """coordinates of v in rref-ish basis list (arbitrary basis): solve over F2"""
    n = len(basis)
    pairs = []
    for i, b in enumerate(basis):
        pairs.append((b, 1 << i))
    red = []
    for w, c in pairs:
        for x, d in red:
            if w ^ x < w:
                w, c = w ^ x, c ^ d
        if w:
            red.append((w, c)); red.sort(key=lambda p: -p[0])
    c = 0
    for x, d in red:
        if v ^ x < v:
            v, c = v ^ x, c ^ d
    assert v == 0, "not in span"
    return c

def complement(sub, n):
    sub = rref(sub); comp = []
    cur = list(sub)
    for i in range(n):
        u = 1 << i
        if not in_span(u, cur):
            cur = rref(cur + [u]); comp.append(u)
    return comp

def graded(L, L0, L1):
    """basis: g_{-1} = compl of L0, g_0 = compl of L1 in L0, g_1 = L1 (depth 1)."""
    n = L.n
    gm = complement(L0, n)
    # complement of L1 inside L0
    cur = rref(L1); g0 = []
    for u in all_elements(rref(L0)):
        if u and not in_span(u, cur):
            cur = rref(cur + [u]); g0.append(u)
    g1 = list(rref(L1))
    B = gm + g0 + g1
    deg = [-1]*len(gm) + [0]*len(g0) + [1]*len(g1)
    m = len(B)
    tabL = [[coords(L.mul(B[i], B[j]), B) for j in range(m)] for i in range(m)]
    tabG = []
    for i in range(m):
        row = []
        for j in range(m):
            c = tabL[i][j]; d = deg[i] + deg[j]
            row.append(sum(((c >> k) & 1) << k for k in range(m) if deg[k] == d))
        tabG.append(row)
    return LieF2(tabL, "L"), LieF2(tabG, "grL"), deg

def inv(L):
    return dict(simple=L.is_simple(), derived=len(L.derived()), der=L.dim_der(),
                nil2=L.num_2nilps() + 1)

def mats(G, deg, a, b):
    """matrices of ad_x : g_a -> g_b? here: action of g_0 elements on g_{-1}"""
    idx0 = [i for i, d in enumerate(deg) if d == 0]
    idxm = [i for i, d in enumerate(deg) if d == a]
    out = []
    for i in idx0:
        M = []
        for j in idxm:
            w = G.tab[i][j]
            M.append([(w >> k) & 1 for k in idxm])   # row j = image of basis j
        out.append(M)
    return out

def matT(M): return [list(r) for r in zip(*M)]
def matmul(A, B):
    n = len(A); m = len(B[0])
    return [[sum(A[i][k] & B[k][j] for k in range(len(B))) & 1 for j in range(m)] for i in range(n)]
def vec(M): return sum(M[i][j] << (i*len(M)+j) for i in range(len(M)) for j in range(len(M)))

def span_of(ms):
    return tuple(sorted(all_elements(rref([vec(M) for M in ms]))))

GL4 = []
def gl4():
    if GL4: return GL4
    for bits in range(1 << 16):
        M = [[(bits >> (4*i+j)) & 1 for j in range(4)] for i in range(4)]
        # invertible over F2?
        r = rref([sum(M[i][j] << j for j in range(4)) for i in range(4)])
        if len(r) == 4:
            GL4.append(M)
    return GL4

def inverse(M):
    n = len(M)
    A = [row[:] + [int(i == j) for j in range(n)] for i, row in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c])
        A[c], A[p] = A[p], A[c]
        for r in range(n):
            if r != c and A[r][c]:
                A[r] = [x ^ y for x, y in zip(A[r], A[c])]
    return [row[n:] for row in A]

def conj_class_rep(ms):
    """canonical form of the 8-dim image subspace under GL(4,F2) conjugation"""
    best = None
    for g in gl4():
        gi = inverse(g)
        s = span_of([matmul(matmul(g, M), gi) for M in ms])
        if best is None or s < best:
            best = s
    return best

# ---------------------------------------------------------- Lebedev's table
TAB = """1&--&--&--&10&--&00&11&--&--&11&--&--&10&--&--&01
7&--&--&--&10&01&00&11&10&10&11&--&--&10&--&--&01
8&--&--&11&--&01&00&11&10&--&--&00&01&--&11&00&--
13&--&00&--&10&--&00&11&--&--&11&--&--&10&11&00&01
14&01&--&11&--&01&--&--&10&--&--&--&01&10&11&00&01"""
ORD = ["00", "01", "10", "11"]
def lebedev(row):
    f = row.split("&"); N = int(f[0]); e = f[1:]
    gens = []
    for g in range(4):
        M = [[0]*4 for _ in range(4)]
        for j in range(4):
            t = e[4*g + j]
            if t != "--":
                M[j][ORD.index(t)] = 1      # row j = image of v_j
        gens.append(M)
    return N, gens   # X1+, X1-, X2+, X2-

def br(A, B):
    # action convention: row vector * matrix; for commutator of operators, use matrices acting on rows
    AB = matmul(A, B); BA = matmul(B, A)
    return [[x ^ y for x, y in zip(r, s)] for r, s in zip(AB, BA)]

def lie_closure(gens):
    rows = rref([vec(M) for M in gens]); basis = [M for M in gens]
    els = list(gens)
    changed = True
    while changed:
        changed = False
        for A in list(els):
            for B in list(els):
                C = br(A, B); v = vec(C)
                if v and not in_span(v, rows):
                    rows = rref(rows + [v]); els.append(C); changed = True
    return els, len(rows)

def main():
    L = build(os.path.join(BASE, "Eick_{16,4}.txt"))
    print("Eick_{16,4} = 16:6", inv(L), flush=True)
    subs0 = all_smallest_codim(L, cmax=4)
    seen = {}
    for U, M in subs0:
        seen.setdefault(tuple(sorted(rref(M))), (U, M))
    subs = list(seen.values())
    print("distinct subalgebras of smallest codim:", len(subs), [L.n - len(M) for _, M in subs], flush=True)
    results = []
    for k, (_, L0) in enumerate(subs):
        L0 = rref(L0)
        L1 = rref([u for u in all_elements(L0) if u and all(in_span(L.mul(u, a), L0) for a in range(1, 1 << L.n) if True and False)] )
        # L_1 = {x in L0 : [x, L] subset L0}
        L1 = rref([u for u in all_elements(L0) if u and all(in_span(L.mul(u, 1 << j), L0) for j in range(L.n))])
        L2 = rref([u for u in all_elements(L1) if u and all(in_span(L.mul(u, 1 << j), L1) for j in range(L.n))])
        print(" sub %d: dims L0 %d, L1 %d, L2 %d" % (k, len(L0), len(L1), len(L2)), flush=True)
        if len(L1) != 4 or L2:
            continue
        LL, G, deg = graded(L, L0, L1)
        ok, msg = G.check_jacobi()
        print("   gr L axioms", msg, inv(G), flush=True)
        print("   L  differs from gr L as a bracket:", LL.tab != G.tab)
        g0idx = [i for i, d in enumerate(deg) if d == 0]
        # g_0 as a Lie algebra
        sub0 = LieF2([[ (sum(((G.tab[i][j] >> k) & 1) << t for t, k in enumerate(g0idx))) for j in g0idx] for i in g0idx], "g0")
        print("   g_0:", inv(sub0), flush=True)
        Am = mats(G, deg, -1, -1); A1 = mats(G, deg, 1, 1)
        results.append((k, Am, A1, G, deg))
    if not results:
        return
    # compare with Lebedev's modules
    reps = {}
    for row in TAB.splitlines():
        N, gens = lebedev(row)
        els, d = lie_closure(gens)
        reps[N] = conj_class_rep(els)
        print("N%d: generated subalgebra of gl(4) has dim %d" % (N, d), flush=True)
    # weighted densities F_lambda over F2, lambda = 0, 1:  from realizVF  g5..g12 acting on g1..g4
    for k, Am, A1, G, deg in results:
        c = conj_class_rep(Am)
        cT = conj_class_rep([matT(M) for M in Am])
        c1 = conj_class_rep(A1)
        hits = [N for N, r in reps.items() if r == c]
        print("sub %d: g_{-1} conj to Lebedev N in %s; transpose (dual) in %s; g_1 in %s; g_1 ~ g_{-1}^* (as images): %s"
              % (k, hits, [N for N, r in reps.items() if r == cT], [N for N, r in reps.items() if r == c1], c1 == cT))

if __name__ == "__main__":
    main()
