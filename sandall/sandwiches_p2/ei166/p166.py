#!/usr/bin/env python3
"""The Cartan prolongs (M, sl(3))_{*,N} inside vect(4;N) over F_2 for the 4-dimensional
sl(3)-modules M of A. Lebedev's table (groups N1, N7, N8) and for the weighted densities,
for N=(1,1,1,1) (and any heights <= 2^k via divided powers).
A vector field is a dict {(exponent tuple, i): 1} over F_2 (set of terms).
Divided powers: x^(a) x^(b) = binom(a+b,a) x^(a+b), d_i x^(a) = x^(a - e_i)."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from css import LieF2, rref, in_span, all_elements
from eick166 import lebedev, lie_closure, TAB, coords

NV = 4
def binom2(a, b):     # binom(a+b, a) mod 2 : no carries
    return 0 if (a & b) else 1

def mono_mul(r, s, H):
    t = tuple(a + b for a, b in zip(r, s))
    if any(t[i] > H[i] for i in range(NV)): return None
    for a, b in zip(r, s):
        if a & b: return None
    return t

def fmul(f, mon, H):
    out = set()
    for (r, i) in f:
        t = mono_mul(r, mon, H)
        if t is not None: out ^= {(t, i)}
    return frozenset(out)

def d(r, i):
    if r[i] == 0: return None
    return tuple(r[k] - (k == i) for k in range(NV))

def bracket(f, g, H):
    out = set()
    for (r, i) in f:
        for (s, j) in g:
            ds = d(s, i)                     # x^r d_i (x^s) d_j
            if ds is not None:
                t = mono_mul(r, ds, H)
                if t is not None: out ^= {(t, j)}
            dr = d(r, j)                     # x^s d_j (x^r) d_i
            if dr is not None:
                t = mono_mul(s, dr, H)
                if t is not None: out ^= {(t, i)}
    return frozenset(out)

def mons(deg, H):
    return [r for r in itertools.product(*[range(h + 1) for h in H]) if sum(r) == deg]

def fields(deg, H):
    return [frozenset({(r, i)}) for r in mons(deg + 1, H) for i in range(NV)]

class Span:
    def __init__(self, vecs, index):
        self.index = index; self.rows = rref([self.tov(v) for v in vecs])
    def tov(self, f):
        v = 0
        for t in f: v ^= 1 << self.index[t]
        return v
    def contains(self, f): return in_span(self.tov(f), self.rows)

def prolong(g0mats, H, maxdeg=8):
    """g_{-1} = d_i ; g_0 = linear fields of the matrices (row k = image of v_k)."""
    gm1 = [frozenset({((0,)*NV, i)}) for i in range(NV)]
    g0 = []
    for M in g0mats:
        f = set()
        for k in range(NV):
            for j in range(NV):
                if M[k][j]: f ^= {(tuple(int(t == k) for t in range(NV)), j)}
        g0.append(frozenset(f))
    comps = [gm1, g0]
    for deg in range(1, maxdeg + 1):
        cand = fields(deg, H)
        if not cand: break
        prev = comps[-1]
        idxp = {t: n for n, t in enumerate(sorted({t for r in mons(deg, H) for t in [(r, i) for i in range(NV)]}))}
        sp = Span(prev, idxp)
        # linear map D -> ([D, d_k] mod prev)_k ; solve kernel by Gaussian elimination over F2
        # build matrix: columns = cand, rows = coordinates of the 4 brackets modulo span(prev)
        def red(f):
            v = sp.tov(f)
            for r in sp.rows:
                if v ^ r < v: v ^= r
            return v
        nb = len(idxp)
        cols = []
        for D in cand:
            v = 0
            for k in range(NV):
                v |= red(bracket(D, gm1[k], H)) << (k * nb)
            cols.append(v)
        # kernel of the map sum c_j cols_j = 0
        pairs = []
        kern = []
        for jdx, c in enumerate(cols):
            v, comb = c, 1 << jdx
            for (w, cw) in pairs:
                if v ^ w < v: v, comb = v ^ w, comb ^ cw
            if v:
                pairs.append((v, comb)); pairs.sort(key=lambda p: -p[0])
            else:
                kern.append(comb)
        kern = rref(kern)
        comp = []
        for comb in kern:
            f = set()
            for jdx in range(len(cand)):
                if (comb >> jdx) & 1: f ^= set(cand[jdx])
            comp.append(frozenset(f))
        if not comp: break
        comps.append(comp)
    return comps

def algebra(comps, H):
    basis = [f for c in comps for f in c]
    allterms = sorted({t for f in basis for t in f})
    index = {t: n for n, t in enumerate(allterms)}
    vecs = []
    for f in basis:
        v = 0
        for t in f: v ^= 1 << index[t]
        vecs.append(v)
    tab = []
    for a in basis:
        row = []
        for b in basis:
            c = bracket(a, b, H); v = 0
            for t in c:
                assert t in index, "not closed"
                v ^= 1 << index[t]
            row.append(coords(v, vecs))
        tab.append(row)
    return LieF2(tab, "prolong")

if __name__ == "__main__":
    H = tuple(int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1111"))
    for row in TAB.splitlines():
        N, gens = lebedev(row)
        els, dim = lie_closure(gens)
        comps = prolong(els[:0] + [M for M in els], H)
        dims = [len(c) for c in comps]
        print("N%d heights %s: dims %s" % (N, H, dims), flush=True)
