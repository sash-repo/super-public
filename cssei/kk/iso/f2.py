"""Small F_2 linear-algebra kit on Python ints (bit i = coordinate i)."""
import json
class Ech:
    """echelon basis keyed by highest bit; reduce() returns the residue"""
    def __init__(s): s.p = {}; s.tag = {}
    def reduce(s, v, t=0):
        while v:
            h = v.bit_length() - 1
            r = s.p.get(h)
            if r is None: return v, t
            v ^= r; t ^= s.tag[h]
        return 0, t
    def add(s, v, t=0):
        v, t = s.reduce(v, t)
        if v: h = v.bit_length() - 1; s.p[h] = v; s.tag[h] = t; return True
        return False
    def __len__(s): return len(s.p)
    def basis(s): return list(s.p.values())
def span(vs):
    E = Ech()
    for v in vs: E.add(v)
    return E
def kernel(imgs):
    """imgs[i] = image of e_i; basis of kernel as ints over e_i"""
    E = Ech(); K = []
    for i, v in enumerate(imgs):
        r, t = E.reduce(v, 1 << i)
        if r: E.add(r, t)
        else: K.append(t)
    return K
class Alg:
    def __init__(s, tab):
        s.tab = tab; s.n = len(tab)
    @classmethod
    def load(c, f):
        d = json.load(open(f)); a = c(d['tab']); a.deg = d.get('deg'); return a
    def br(s, a, b):
        t = s.tab; out = 0; i = 0
        while a:
            if a & 1:
                row = t[i]; bb = b; j = 0
                while bb:
                    if bb & 1: out ^= row[j]
                    bb >>= 1; j += 1
            a >>= 1; i += 1
        return out
    def ad(s, x):
        """columns of ad x"""
        return [s.br(x, 1 << j) for j in range(s.n)]
def bits(v):
    i = 0
    while v:
        if v & 1: yield i
        v >>= 1; i += 1
def apply(M, v):
    out = 0
    for j in bits(v): out ^= M[j]
    return out
def mul(A, B):   # A∘B, columns
    return [apply(A, c) for c in B]
def flat(M, n):
    v = 0
    for j, c in enumerate(M): v |= c << (j * n)
    return v
def rank(cols):
    return len(span(cols))
def kernel_mod(imgs, V):
    """kernel of e_i -> imgs[i] mod span V (V an Ech): seed the echelon with V (tag 0)"""
    E = Ech()
    for v in V.basis(): E.add(v, 0)
    K = []
    for i, v in enumerate(imgs):
        r, t = E.reduce(v, 1 << i)
        if r: E.add(r, t)
        else: K.append(t)
    return K
def full_reduce(E, v):
    """canonical representative of v mod span(E): clear every pivot bit, top down"""
    for h in sorted(E.p, reverse=True):
        if v >> h & 1: v ^= E.p[h]
    return v
