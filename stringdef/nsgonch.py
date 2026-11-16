"""Cohomology with trivial coefficients of the positive parts of witt (L_k) and of the
Neveu-Schwarz / Ramond superalgebras, graded by weight. Exact arithmetic via exactlin.
Cochains: S(Pi g^*) = Lambda(even duals) (x) S(odd duals). Work with chains (homology,
dual, same dims): C_q = Lambda^q in super sense of g, d(x1..xq) = sum_{i<j} +-[xi,xj] x1..^..^..xq.
"""
import sys, itertools
from fractions import Fraction as F
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank

def algebra(kind, W):
    # basis elements: (name, weight(as Fraction), parity)
    B = []
    if kind == "witt1":
        B = [("L", F(n), 0) for n in range(1, W+1)]
    elif kind.startswith("wittk"):
        k = int(kind[5:]); B = [("L", F(n), 0) for n in range(k, W+1)]
    elif kind == "ns":
        B = [("L", F(n), 0) for n in range(1, W+1)] + [("G", F(2*r+1, 2), 1) for r in range(0, W) if F(2*r+1,2) <= W]
    elif kind == "ns2":   # L_n n>=2, G_r r>=3/2
        B = [("L", F(n), 0) for n in range(2, W+1)] + [("G", F(2*r+1, 2), 1) for r in range(1, W) if F(2*r+1,2) <= W]
    elif kind.startswith("nsd"):   # NS elements of degree >= m/2
        m = int(kind[3:]); lo = F(m, 2)
        B = [("L", F(n), 0) for n in range(1, W+1) if F(n) >= lo] + [("G", F(2*r+1, 2), 1) for r in range(0, W) if lo <= F(2*r+1,2) <= W]
    elif kind.startswith("ramd"):  # Ramond elements of degree >= m
        m = int(kind[4:])
        B = [("L", F(n), 0) for n in range(m, W+1)] + [("G", F(n), 1) for n in range(m, W+1)]
    elif kind == "ram":   # L_n n>=1, G_n n>=1 (Ramond)
        B = [("L", F(n), 0) for n in range(1, W+1)] + [("G", F(n), 1) for n in range(1, W+1)]
    idx = {(b[0], b[1]): i for i, b in enumerate(B)}
    def br(i, j):
        (a, m, _), (b, n, _) = B[i], B[j]
        if a == "L" and b == "L": res = [("L", m+n, m-n)]
        elif a == "L" and b == "G": res = [("G", m+n, m/2 - n)]
        elif a == "G" and b == "L": res = [("G", m+n, -(n/2 - m))]
        else: res = [("L", m+n, F(2))]
        out = {}
        for (c, w, co) in res:
            if co != 0 and (c, w) in idx: out[idx[(c, w)]] = co
        return out
    return B, br

def chains(B, q, w):
    """super-exterior monomials: sorted index tuples, even indices distinct, odd may repeat."""
    n = len(B); res = []
    def rec(start, left, wt, cur):
        if left == 0:
            if wt == w: res.append(tuple(cur))
            return
        for i in range(start, n):
            nw = wt + B[i][1]
            if nw > w: continue
            nxt = i if B[i][2] == 1 else i + 1
            cur.append(i); rec(nxt, left-1, nw, cur); cur.pop()
    rec(0, q, F(0), [])
    return res

def normal(mono, B):
    """sort a list of indices with super signs: swapping x,y gives -(-1)^{p(x)p(y)} ...
    In Pi-shifted picture: chain space = S(Pi g): elements Pi x have parity p(x)+1.
    swap sign = (-1)^{(px+1)(py+1)}; repeated even -> 0."""
    m = list(mono); s = 1
    for a in range(len(m)):
        for b in range(len(m)-1-a):
            if m[b] > m[b+1]:
                pa, pb = B[m[b]][2]+1, B[m[b+1]][2]+1
                if (pa*pb) % 2: s = -s
                m[b], m[b+1] = m[b+1], m[b]
    for a in range(len(m)-1):
        if m[a] == m[a+1] and B[m[a]][2] == 0: return None, 0
    return tuple(m), s

def dmat(B, br, q, w):
    src = chains(B, q, w); tgt = chains(B, q-1, w)
    ti = {t: i for i, t in enumerate(tgt)}
    rows = []
    for m in src:
        v = {}
        for i in range(q):
            for j in range(i+1, q):
                c = br(m[i], m[j])
                if not c: continue
                # move xi, xj to front: sign from passing Pi-parities
                rest = [m[t] for t in range(q) if t != i and t != j]
                pi = lambda t: B[m[t]][2] + 1
                s = 1
                # move m[i] to position 0
                for t in range(i):
                    if (pi(i)*pi(t)) % 2: s = -s
                for t in range(j):
                    if t == i: continue
                    if (pi(j)*pi(t)) % 2: s = -s
                # d(Pi x Pi y ...) = (-1)^{p(Pi x)} Pi[x,y] ...  (one consistent choice)
                if pi(i) % 2: s = -s
                for k, co in c.items():
                    nm, s2 = normal([k] + rest, B)
                    if nm is None: continue
                    v[ti[nm]] = v.get(ti[nm], 0) + s*s2*co
        rows.append([v.get(t, 0) for t in range(len(tgt))])
    return rows, len(src), len(tgt)

def homology(kind, W, Q):
    B, br = algebra(kind, W)
    # check d^2=0 lightly by ranks consistency is not proof; do explicit check for small
    out = {}
    ws = sorted({F(n, 2) for n in range(0, 2*W+1)})
    for w in ws:
        for q in range(0, Q+1):
            def rk(q):
                if q <= 0: return 0
                M, a, b = dmat(B, br, q, w)
                if a == 0 or b == 0: return 0
                return rank(M)
            dimq = len(chains(B, q, w))
            h = dimq - rk(q) - rk(q+1)
            if h: out[(q, w)] = h
    return out

def d2check(kind, W, q, w):
    B, br = algebra(kind, W)
    M1, a, b = dmat(B, br, q, w); M2, b2, c = dmat(B, br, q-1, w)
    for r in M1:
        for k in range(c):
            if sum(r[t]*M2[t][k] for t in range(b)) != 0: return False
    return True

if __name__ == "__main__":
    kind = sys.argv[1]; W = int(sys.argv[2]); Q = int(sys.argv[3])
    for q in range(2, Q+1):
        for w in [F(n,2) for n in range(1, 2*min(W, q+6)+1)]:
            assert d2check(kind, W, q+1, w), (q, w)
    res = homology(kind, W, Q)
    for q in range(Q+1):
        print(q, [(str(w), h) for (qq, w), h in sorted(res.items()) if qq == q])
