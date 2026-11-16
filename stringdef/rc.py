#!/usr/bin/env python3
"""Structure functions (analogs of the curvature tensor) H^2(g_-; g)_d of a
distribution on C^{1|n} given by a 1-form, after BGLS (arXiv:1010.4480).

Coordinates: t even (index 0), odd x_1..x_n (indices 1..n).  The distribution D is
given by its F-frame; g = all polynomial vector fields X with alpha([X,L])=0 for the
frame L (the maximal Tanaka prolong of the symbol), graded by deg t = 2, deg x = 1.
g_- = its negative part.  H^2 computed per degree d and parity of the cochain.
Cocycle/coboundary written via the CE differential with module = g restricted to g_-.
Sign-free trick: the CE differential is tested against the Jacobi identity of
g_- (+) eps*g (see self-test)."""
import sys, itertools
from fractions import Fraction as Fr
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, nullspace

# ---- functions: dict {(a, S): c}, t^a x_S, S sorted tuple of odd indices
def srt(S):
    S = list(S); s = 1
    for i in range(len(S)):
        for j in range(len(S)-1-i):
            if S[j] == S[j+1]: return None, 0
            if S[j] > S[j+1]: S[j], S[j+1] = S[j+1], S[j]; s = -s
    for i in range(len(S)-1):
        if S[i] == S[i+1]: return None, 0
    return tuple(S), s
def fadd(f, g, c=1):
    h = dict(f)
    for k, v in g.items():
        h[k] = h.get(k, 0) + c*v
        if h[k] == 0: del h[k]
    return h
def fmul(f, g):
    h = {}
    for (a, S), u in f.items():
        for (b, T), v in g.items():
            U, s = srt(S+T)
            if s:
                k = (a+b, U); h[k] = h.get(k, 0) + s*u*v
                if h[k] == 0: del h[k]
    return h
def fpar(f):
    ps = {len(S) % 2 for (a, S) in f}
    assert len(ps) <= 1; return ps.pop() if ps else 0
def dcoord(j, f):
    h = {}
    for (a, S), c in f.items():
        if j == 0:
            if a: h[(a-1, S)] = h.get((a-1, S), 0) + a*c
        elif j in S:
            p = S.index(j); T = S[:p]+S[p+1:]
            h[(a, T)] = h.get((a, T), 0) + (-1)**p*c
    return {k: v for k, v in h.items() if v}
# ---- vector fields: list of functions, component k = coefficient of d_k
def vpar(X, n):
    ps = set()
    for k in range(n+1):
        if X[k]: ps.add((fpar(X[k]) + (k > 0)) % 2)
    assert len(ps) <= 1; return ps.pop() if ps else 0
def vapply(X, f, n):
    h = {}
    for j in range(n+1):
        if X[j]: h = fadd(h, fmul(X[j], dcoord(j, f)))
    return h
def vbr(X, Y, n):
    s = (-1)**(vpar(X, n)*vpar(Y, n))
    return [fadd(vapply(X, Y[k], n), vapply(Y, X[k], n), -s) for k in range(n+1)]
def vzero(n): return [{} for _ in range(n+1)]
def vlin(coeffs, basis, n):
    Z = vzero(n)
    for c, B in zip(coeffs, basis):
        if c: Z = [fadd(Z[k], B[k], c) for k in range(n+1)]
    return Z

WT = {}   # weight of odd coordinate i (default 1); set WT[i]=0 for zeta of weight 0
def monos(n, w):  # monomials of weight w (deg t=2, deg x_i = WT.get(i,1))
    out = []
    for r in range(0, n+1):
        for S in itertools.combinations(range(1, n+1), r):
            ws = sum(WT.get(i, 1) for i in S)
            if (w - ws) >= 0 and (w - ws) % 2 == 0:
                out.append(((w-ws)//2, S))
    return out
def vfields(n, d):  # all monomial vector fields of degree d
    out = []
    for k in range(n+1):
        wk = 2 if k == 0 else WT.get(k, 1)
        for m in monos(n, d + wk):
            X = vzero(n); X[k] = {m: Fr(1)}; out.append(X)
    return out
def vkey(X, n):
    return [((k,) + m, c) for k in range(n+1) for m, c in X[k].items()]

class Distr:
    """frame: list of vector fields; alpha(Y) linear functional returning a function."""
    def __init__(self, n, frame, alpha):
        self.n, self.frame, self.alpha = n, frame, alpha
        self.g = {}
    def comp(self, d):
        if d in self.g: return self.g[d]
        n = self.n; V = vfields(n, d)
        # condition alpha([X, L]) = 0 for every frame L; linear in coefficients
        rows = {}
        for i, X in enumerate(V):
            for li, L in enumerate(self.frame):
                a = self.alpha(vbr(X, L, n))
                for m, c in a.items():
                    rows.setdefault((li, m), {})[i] = c
        M = [[r.get(i, 0) for i in range(len(V))] for r in rows.values()]
        ns = nullspace(M) if M else [[Fr(int(i == j)) for i in range(len(V))] for j in range(len(V))]
        B = [vlin(v, V, n) for v in ns]
        # split by parity
        B = [b for b in B if vpar(b, n) == 0] + [b for b in B if vpar(b, n) == 1] \
            if all(len({vpar(b, n)}) for b in B) else B
        self.g[d] = B
        return B

def coords(Y, basis, n):
    """coordinates of Y in basis (exact solve)."""
    keys = {}
    cols = [dict(vkey(b, n)) for b in basis]
    y = dict(vkey(Y, n))
    allk = set(y) | set().union(*cols) if cols else set(y)
    allk = sorted(allk)
    M = [[c.get(k, 0) for c in cols] + [y.get(k, 0)] for k in allk]
    from exactlin import solve
    A = [r[:-1] for r in M]; b = [r[-1] for r in M]
    x = solve(A, b)
    assert x is not None, "not in span"
    return x

def H2(D, d, depth=2):
    """dims of H^2(g_-; g)_d split by cochain parity, plus Z^2, B^2 data."""
    n = D.n
    gm = [(k, b) for k in range(-depth, 0) for b in D.comp(k)]
    par = [vpar(b, n) for _, b in gm]
    N = len(gm)
    # basis of Lambda^q g_- (super): sorted index tuples, repeats only for odd
    def wedge(q):
        out = []
        for I in itertools.combinations_with_replacement(range(N), q):
            if all(not (I[a] == I[a+1] and par[I[a]] == 0) for a in range(q-1)):
                out.append(I)
        return out
    def target(I, deg): return deg + sum(gm[i][0] for i in I)
    res = {}
    for cp in (0, 1):
        # cochain spaces C^q_d: for each wedge I, a basis vector of g_{target} of parity
        def cspace(q):
            sp = []
            for I in wedge(q):
                tg = target(I, d)
                if tg < -depth: continue
                for j, b in enumerate(D.comp(tg)):
                    if (vpar(b, n) - sum(par[i] for i in I) - cp) % 2 == 0:
                        sp.append((I, tg, j))
            return sp
        C1, C2, C3 = cspace(1), cspace(2), cspace(3)
        res[cp] = (C1, C2, C3)
    return gm, par, res


def flat(Y, n): return dict(vkey(Y, n))

def cohom(D, d, depth=2, check=False, verbose=False):
    n = D.n
    gm, par, res = H2(D, d, depth)
    N = len(gm)
    xs = [b for _, b in gm]
    brk = {}
    for i in range(N):
        for j in range(N):
            brk[i, j] = coords(vbr(xs[i], xs[j], n), xs, n)
    out = {}
    for cp in (0, 1):
        C1, C2, C3 = res[cp]
        idx2 = {e: k for k, e in enumerate(C2)}
        def ev1(phi, i):  # phi vector over C1 -> vector field
            Z = vzero(n)
            for k, (I, tg, j) in enumerate(C1):
                if phi[k] and I == (i,):
                    Z = [fadd(Z[a], D.comp(tg)[j][a], phi[k]) for a in range(n+1)]
            return Z
        def ev1lin(phi, coeff):
            Z = vzero(n)
            for i, c in enumerate(coeff):
                if c:
                    E = ev1(phi, i); Z = [fadd(Z[a], E[a], c) for a in range(n+1)]
            return Z
        def ev2(c, i, j):
            s = 1
            if i > j: i, j = j, i; s = -(-1)**(par[i]*par[j])
            if i == j and par[i] == 0: return vzero(n)
            Z = vzero(n)
            for k, (I, tg, jj) in enumerate(C2):
                if c[k] and I == (i, j):
                    Z = [fadd(Z[a], D.comp(tg)[jj][a], s*c[k]) for a in range(n+1)]
            return Z
        def ev2lin1(c, coeff, j):  # c(sum coeff_i x_i, x_j)
            Z = vzero(n)
            for i, a0 in enumerate(coeff):
                if a0:
                    E = ev2(c, i, j); Z = [fadd(Z[a], E[a], a0) for a in range(n+1)]
            return Z
        def act(i, Y, ypar):  # x_i . Y with cochain sign handled outside
            return vbr(xs[i], Y, n)
        def d1(phi, i, j):
            pi_, pj = par[i], par[j]
            A = vbr(xs[i], ev1(phi, j), n); B = vbr(xs[j], ev1(phi, i), n)
            C = ev1lin(phi, brk[i, j])
            s1 = (-1)**(pi_*cp); s2 = -(-1)**(pj*(pi_+cp))
            return [fadd(fadd(fadd({}, A[a], s1), B[a], s2), C[a], -1) for a in range(n+1)]
        def d2(c, i, j, k):
            p = par
            t1 = vbr(xs[i], ev2(c, j, k), n); s1 = (-1)**(p[i]*cp)
            t2 = vbr(xs[j], ev2(c, i, k), n); s2 = -(-1)**(p[j]*(p[i]+cp))
            t3 = vbr(xs[k], ev2(c, i, j), n); s3 = (-1)**(p[k]*(p[i]+p[j]+cp))
            t4 = ev2lin1(c, brk[i, j], k); s4 = -1
            t5 = ev2lin1(c, brk[i, k], j); s5 = (-1)**(p[j]*p[k])
            t6 = ev2lin1(c, brk[j, k], i); s6 = -(-1)**(p[i]*(p[j]+p[k]))
            Z = vzero(n)
            for T, s in ((t1, s1), (t2, s2), (t3, s3), (t4, s4), (t5, s5), (t6, s6)):
                Z = [fadd(Z[a], T[a], s) for a in range(n+1)]
            return Z
        # matrix of d1 : C1 -> C2 (in C2 coordinates)
        rows1 = []
        for k in range(len(C1)):
            phi = [0]*len(C1); phi[k] = 1
            col = [Fr(0)]*len(C2)
            for (I, tg, jj) in set((e[0], e[1], None) for e in C2):
                i, j = I
                Y = d1(phi, i, j)
                if all(not Y[a] for a in range(n+1)): continue
                basis = [D.comp(tg)[q] for q in range(len(D.comp(tg)))]
                x = coords(Y, basis, n)
                for q, v in enumerate(x):
                    if v:
                        col[idx2[(I, tg, q)]] += v
            rows1.append(col)
        rk1 = rank(rows1) if rows1 and C2 else 0
        # Z2 = kernel of d2 : C2 -> functions on wedge3
        trip = [I for I in itertools.combinations_with_replacement(range(N), 3)
                if all(not (I[a] == I[a+1] and par[I[a]] == 0) for a in range(2))]
        cols = []
        for k in range(len(C2)):
            c = [0]*len(C2); c[k] = 1
            col = {}
            for I in trip:
                for kk, v in flat(d2(c, *I), n).items():
                    col[(I, kk)] = v
            cols.append(col)
        keys = sorted(set().union(*cols)) if cols else []
        M = [[col.get(kk, 0) for col in cols] for kk in keys]
        rkd2 = rank(M) if M else 0
        z2 = len(C2) - rkd2
        if check and rows1 and C2:
            # d2 o d1 = 0
            for k in range(len(C1)):
                c = rows1[k]
                for I in trip:
                    Y = d2(c, *I)
                    assert all(not Y[a] for a in range(n+1)), ("d^2 != 0", cp, k, I)
        out[cp] = dict(C1=len(C1), C2=len(C2), Z2=z2, B2=rk1, H2=z2-rk1)
    return out
