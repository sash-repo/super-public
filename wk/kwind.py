#!/usr/bin/env python3
"""kwind.py TAB.json [...] -- for a simple Lie (super)algebra g over F_p (p=2,3) given by
structure constants: the derived algebra g' (if g is not perfect), its minimal p-envelope
L in Der g' (the span of ad g' and of the p^k-th powers of its even basis elements), the
index of L (the least superdimension of the stabilizer of chi in L_ev^*, chi(L_od)=0),
whether the generic stabilizer is a torus, and the number p^{(dim L_ev-s_ev)/2} 2^{ceil((dim L_od-s_od)/2)}.
Generic chi: random points of F_{2^16} / F_{3^10}; the minimum over TRIALS is reported.
Only [L,L] in ad g' is used: for derivations D, E in L, [D,E] is inner (induction on the
p-power level; Lemma 9.1.1 of the text), so chi enters through g' only.  Output: one json line
per algebra.  This is the computation behind Table (16), Statement 9.2.1 and Remark 9.2.3 of the text;
the number reported is N(g) of formulas (14), (15)."""
import sys, json, math, time
import numpy as np
import galois

TRIALS = 3

def load(path):
    T = json.load(open(path)); p = T.get('p', 2); n = T['dim']
    C = np.zeros((n, n, n), dtype=np.int64)
    if 'tab' in T:
        for i in range(n):
            for j in range(n):
                m = T['tab'][i][j]; k = 0
                while m:
                    if m & 1: C[i, j, k] = 1
                    m >>= 1; k += 1
    else:
        for i, j, k, c in T['C']: C[i, j, k] = c % p
    par = T.get('par', [0] * n)
    return T.get('name', path), p, C, np.array(par)

def rref_mod(A, p):
    """rows of A mod p -> (R, piv, Tm) with R = Tm A in RREF (R has rank rows)."""
    A = np.array(A, dtype=np.int64) % p; n, m = A.shape
    M = np.concatenate([A, np.eye(n, dtype=np.int64)], axis=1); piv = []; r = 0
    for c in range(m):
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0: continue
        i = r + nz[0]
        if i != r: M[[r, i]] = M[[i, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p if p > 2 else M[r]
        col = M[:, c].copy(); col[r] = 0; nzr = np.nonzero(col)[0]
        if len(nzr): M[nzr] = (M[nzr] - np.outer(col[nzr], M[r])) % p
        piv.append(c); r += 1
        if r == n: break
    return M[:r, :m], piv, M[:r, m:]

class Span:
    """incremental span of vectors mod p (for the p-envelope)"""
    def __init__(self, p, m): self.p = p; self.R = np.zeros((0, m), dtype=np.int64); self.piv = []
    def reduce(self, v):
        v = v % self.p
        for t, c in enumerate(self.piv):
            if v[c]: v = (v - v[c] * self.R[t]) % self.p
        return v
    def add(self, v):
        v = self.reduce(v); nz = np.nonzero(v)[0]
        if len(nz) == 0: return False
        c = nz[0]; v = (v * pow(int(v[c]), self.p - 2, self.p)) % self.p if self.p > 2 else v
        if len(self.piv):
            col = self.R[:, c].copy(); nzr = np.nonzero(col)[0]
            if len(nzr): self.R[nzr] = (self.R[nzr] - np.outer(col[nzr], v)) % self.p
        self.R = np.vstack([self.R, v]); self.piv.append(c); return True

def nullspace_mod(A, p):
    """vectors z (rows) with z A = 0, i.e. left kernel of A, mod p"""
    A = np.array(A, dtype=np.int64) % p; n = A.shape[0]
    R, piv, Tm = rref_mod(A, p)
    # rows of Tm span the row-operations; left kernel = rows of the full transform beyond rank
    M = np.concatenate([A, np.eye(n, dtype=np.int64)], axis=1)
    R2, piv2, _ = rref_mod(M, p)
    return np.array([r[A.shape[1]:] for r in R2 if not r[:A.shape[1]].any()], dtype=np.int64)

def derived(C, par, p):
    n = C.shape[0]
    R, piv, _ = rref_mod(C.reshape(n * n, n), p)
    if len(piv) == n: return C, par, False
    B = R  # basis of g' in old coords (rows)
    k = len(B); Rb, pb, Tb = rref_mod(B, p)
    def co(v):
        v = v % p; c = np.zeros(k, dtype=np.int64)
        for t, cc in enumerate(pb):
            if v[cc]: c = (c + v[cc] * Tb[t]) % p; v = (v - v[cc] * Rb[t]) % p
        assert not v.any(); return c
    X = np.tensordot(B, C, axes=([1], [0])) % p            # X[a,j,k]
    br = np.tensordot(X, B, axes=([1], [1])).transpose(0, 2, 1) % p   # br[a,b,k]
    C2 = np.array([[co(br[a, b]) for b in range(k)] for a in range(k)])
    par2 = np.array([int(par[np.nonzero(r)[0][0]]) for r in B])
    return C2, par2, True

def analyse(path):
    t0 = time.time()
    name, p, C, par = load(path)
    out = {'name': name, 'p': p, 'dim0': int(C.shape[0])}
    while True:
        C, par, changed = derived(C, par, p)
        if not changed: break
        out.setdefault('derived', []).append(int(C.shape[0]))
    # quotient by the centre (once suffices for a perfect algebra whose quotient is simple)
    n = C.shape[0]
    adm = np.array([C[i].T.reshape(-1) % p for i in range(n)])
    Z = nullspace_mod(adm, p)
    if len(Z):
        out['centre'] = len(Z)
        Rz, pz, _ = rref_mod(Z, p)
        keep = [k for k in range(n) if k not in pz]       # complement: coordinates not pivots of Z
        def red(v):
            v = v % p
            for t, c in enumerate(pz):
                if v[c]: v = (v - v[c] * Rz[t]) % p
            return v
        m = len(keep)
        C2 = np.zeros((m, m, m), dtype=np.int64)
        for a, i in enumerate(keep):
            for b, j in enumerate(keep):
                C2[a, b] = red(C[i, j])[keep]
        C = C2; par = par[keep]
    n = C.shape[0]; out['dim'] = n; out['sdim'] = [int((par == 0).sum()), int((par == 1).sum())]
    ad = [C[i].T.copy() % p for i in range(n)]          # (ad e_i)[:, j] = [e_i, e_j]
    # centre
    Rz, pz, _ = rref_mod(np.array([a.reshape(-1) for a in ad]), p)
    out['centreless'] = len(pz) == n
    # p-envelope: powers of even elements, iterated
    S = Span(p, n * n)
    for a in ad: S.add(a.reshape(-1))
    extra = []; queue = [ad[i] for i in range(n) if par[i] == 0]
    while queue:
        X = queue.pop()
        Y = np.linalg.matrix_power(X, p) % p if p == 2 else (X @ X % p) @ X % p
        if S.add(Y.reshape(-1)): extra.append(Y); queue.append(Y)
    out['dimL'] = n + len(extra); out['restricted'] = not extra
    Lmat = ad + extra; Lpar = list(par) + [0] * len(extra); N = len(Lmat)
    # coordinates in ad g' of a matrix that is inner
    Ra, pa, Ta = rref_mod(np.array([a.reshape(-1) for a in ad]), p)
    def inner(M):
        v = M.reshape(-1) % p; c = np.zeros(n, dtype=np.int64)
        for t, cc in enumerate(pa):
            if v[cc]: c = (c + v[cc] * Ta[t]) % p; v = (v - v[cc] * Ra[t]) % p
        assert not v.any(), '[D,E] not inner'
        return c
    T = np.zeros((N, N, n), dtype=np.int64); T[:n, :n] = C
    for a, D in enumerate(extra):
        A = n + a
        for j in range(n):
            T[A, j] = D[:, j] % p; T[j, A] = (-D[:, j]) % p
        for b, E in enumerate(extra):
            T[A, n + b] = inner(D @ E - E @ D)
    F = galois.GF(2 ** 16) if p == 2 else galois.GF(3 ** 10)
    rng = np.random.default_rng(1773)
    best = None
    Tf = F(T.reshape(N * N, n) % p)
    for trial in range(TRIALS):
        chi = F(rng.integers(0, F.order, n)); chi[np.array(par) == 1] = 0
        B = (Tf @ chi).reshape(N, N)
        ev = [i for i in range(N) if Lpar[i] == 0]; od = [i for i in range(N) if Lpar[i] == 1]
        Bev = B[np.ix_(ev, ev)]; s_ev = len(ev) - int(np.linalg.matrix_rank(Bev))
        s_od = len(od) - int(np.linalg.matrix_rank(B[np.ix_(od, od)])) if od else 0
        if best is None or (s_ev, s_od) < best[:2]:
            # toral test of the even stabilizer
            K = Bev.null_space()                     # rows: vectors in L_ev coords
            Kfull = F.Zeros((K.shape[0], N)); Kfull[:, ev] = K
            ab = True
            if K.shape[0] > 1:
                br = (Kfull @ Tf.reshape(N, N * n)).reshape(K.shape[0], N, n)
                br = np.einsum('aik,bi->abk', br, Kfull) if False else None
                for x in range(K.shape[0]):
                    for y in range(x + 1, K.shape[0]):
                        v = (Kfull[x] @ Tf.reshape(N, N * n)).reshape(N, n)
                        if np.any(Kfull[y] @ v): ab = False; break
                    if not ab: break
            Lf = [F(M % p) for M in Lmat]
            pw = []
            for x in range(K.shape[0]):
                X = F.Zeros((n, n))
                for i in np.nonzero(Kfull[x])[0]: X = X + Kfull[x][i] * Lf[i]
                Y = X
                for _ in range(p - 1): Y = Y @ X
                pw.append(Y.reshape(-1))
            inj = int(np.linalg.matrix_rank(F(np.array(pw)))) == K.shape[0] if pw else True
            best = (s_ev, s_od, bool(ab and inj), bool(ab))
    s_ev, s_od, toral, abel = best
    Lev = sum(1 for x in Lpar if x == 0); Lod = N - Lev
    out.update({'sdimL': [Lev, Lod], 'ind': [s_ev, s_od], 'toral_stab': toral, 'abelian_stab': abel})
    e0 = (Lev - s_ev) // 2; e1 = math.ceil((Lod - s_od) / 2)
    out['bound'] = '%d^%d' % (p, e0) + (' * 2^%d' % e1 if Lod else '')
    out['secs'] = round(time.time() - t0, 1)
    return out

if __name__ == '__main__':
    for f in sys.argv[1:]:
        print(json.dumps(analyse(f)), flush=True)
