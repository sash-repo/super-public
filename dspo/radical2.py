#!/usr/bin/env python3
"""
Statement 3.2.6 (m = 5, 6, 7): a faster version of radical.py: numpy arithmetic mod a prime
p < 2^25 (so products and d-term sums fit in int64).

Radical: N' = {y | tr(ad_y b) = 0 for b in a random sample of the unital
associative envelope A of ad(g)} contains the nilradical; N' is then
CERTIFIED to be a solvable ideal (ideal property + derived series reaching 0),
so g/N' is a quotient by a solvable ideal; iterate.  When N' = 0 the quotient
s has no nonzero nilpotent ideal, hence no solvable one: s = g/r.
Then s is described: sdim, dim s' , dim of the centroid (commutant of ad_s,
from the commutant of two random elements: an upper bound that is exact when
it is 1), and, for s_od = 0, the rank (dim of the centralizer of a random
element).
Usage: python3 radical2.py 5 6 7
"""
import sys, random
import numpy as np
from fractions import Fraction
from po import Po, Quot, homologic
from porun import type_reps

P = 33554393
rnd = np.random.default_rng(5)

def rref(M):
    """rref mod P of an int64 matrix; returns (R, pivots)."""
    M = M.copy() % P
    r = 0; piv = []
    rows, cols = M.shape
    for c in range(cols):
        if r == rows: break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0: continue
        k = r + nz[0]
        if k != r: M[[r, k]] = M[[k, r]]
        inv = pow(int(M[r, c]), -1, P)
        M[r] = (M[r] * inv) % P
        col = M[:, c].copy(); col[r] = 0
        nzr = np.nonzero(col)[0]
        if len(nzr):
            M[nzr] = (M[nzr] - np.outer(col[nzr], M[r]) % P) % P
        piv.append(c); r += 1
    return M[:r], piv

def rank(M):
    return len(rref(M)[1]) if M.size else 0

def nullspace(M, n):
    if M.shape[0] == 0:
        return np.eye(n, dtype=np.int64)
    R, piv = rref(M)
    free = [c for c in range(n) if c not in set(piv)]
    out = np.zeros((len(free), n), dtype=np.int64)
    for a, f in enumerate(free):
        out[a, f] = 1
        for i, pc in enumerate(piv):
            out[a, pc] = (-R[i, f]) % P
    return out

def matmul(A, B):
    # entries < 2^25, inner dim <= 128 -> sum < 2^57
    return (A @ B) % P

class Alg:
    def __init__(self, T, par):
        # T[i, j, :] = coordinates of [e_i, e_j]
        self.T = T % P; self.par = list(par); self.d = len(par)
        self.ad = np.transpose(self.T, (0, 2, 1)).copy()   # ad[i][t][j]
    def brk(self, u, v):
        return np.einsum('i,j,ijt->t', u, v, self.T) % P if False else \
            matmul(matmul(u[None, :], self.T.reshape(self.d, -1)).reshape(self.d, self.d).T @ v[:, None] % P, np.eye(1, dtype=np.int64)).ravel() % P
    def span_br(self, U, V):
        d = self.d
        # all [u, v]: (U T) gives for each u a d x d matrix M_u[j, t]; then V M_u
        out = []
        UT = matmul(U, self.T.reshape(d, d * d)).reshape(len(U), d, d)
        for k in range(len(U)):
            out.append(matmul(V, UT[k]))
        W = np.vstack(out) if out else np.zeros((0, d), dtype=np.int64)
        R, piv = rref(W)
        return R
    def is_ideal(self, N):
        full = np.eye(self.d, dtype=np.int64)
        B = self.span_br(full, N)
        return rank(np.vstack([N, B])) == rank(N)
    def solvable(self, N):
        cur = rref(N)[0]
        for _ in range(40):
            if len(cur) == 0: return True
            nxt = self.span_br(cur, cur)
            if len(nxt) == len(cur): return False
            cur = nxt
        return False
    def rand_elt(self):
        return rnd.integers(0, P, self.d, dtype=np.int64)
    def ad_of(self, y):
        return matmul(y[None, :], self.ad.reshape(self.d, -1)).reshape(self.d, self.d)
    def sample_env(self, per_len=None, maxlen=6):
        d = self.d
        per_len = per_len or (d + 8)
        bs = [np.eye(d, dtype=np.int64)]
        for L in range(1, maxlen + 1):
            for _ in range(per_len):
                b = self.ad_of(self.rand_elt())
                for _ in range(L - 1):
                    b = matmul(b, self.ad_of(self.rand_elt()))
                bs.append(b)
        return bs
    def nil_candidate(self):
        d = self.d
        Af = self.ad.reshape(d, d * d)
        rows = [matmul(Af, b.T.reshape(d * d, 1)).ravel() for b in self.sample_env()]
        return nullspace(np.array(rows, dtype=np.int64), d)
    def quotient(self, N):
        d = self.d
        Nr = rref(N)[0]
        piv = rref(Nr)[1]
        comp = [i for i in range(d) if i not in set(piv)]
        Bm = np.vstack([Nr, np.eye(d, dtype=np.int64)[comp]])   # rows = new basis
        # coordinates: v = c Bm  ->  c = v Bm^{-1}
        Binv = inv(Bm)
        k = len(Nr); nd = len(comp)
        T2 = np.zeros((nd, nd, nd), dtype=np.int64)
        for a, i in enumerate(comp):
            for b, j in enumerate(comp):
                co = matmul(self.T[i, j][None, :], Binv).ravel()
                T2[a, b] = co[k:]
        return Alg(T2, [self.par[i] for i in comp])
    def sdim(self):
        return (self.par.count(0), self.par.count(1))
    def derived_dim(self):
        I = np.eye(self.d, dtype=np.int64)
        return len(self.span_br(I, I))
    def centroid_bound(self):
        d = self.d
        rows = []
        for _ in range(2):
            A = self.ad_of(self.rand_elt())
            # X A - A X = 0 ; unknown X flattened row-major
            K1 = np.kron(np.eye(d, dtype=np.int64), A.T)   # vec(XA)
            K2 = np.kron(A, np.eye(d, dtype=np.int64))     # vec(AX)
            rows.append((K1 - K2) % P)
        return d * d - rank(np.vstack(rows))
    def toral_rank(self):
        A = self.ad_of(self.rand_elt())
        # generalized 0-eigenspace dim of a random ad = rank of the Lie algebra
        M = A.copy()
        for _ in range(self.d): M = matmul(M, A)
        return self.d - rank(M)

def inv(B):
    n = len(B)
    R, piv = rref(np.hstack([B, np.eye(n, dtype=np.int64)]))
    assert piv[:n] == list(range(n))
    return R[:, n:]

def radical_quotient(alg):
    steps = []
    cur = alg
    while cur.d:
        N = cur.nil_candidate()
        if len(N) == 0:
            return cur, steps
        assert cur.is_ideal(N), "candidate not an ideal"
        assert cur.solvable(N), "candidate not solvable"
        steps.append(len(N))
        cur = cur.quotient(N)
    return cur, steps

def modp(c):
    c = Fraction(c)
    return c.numerator * pow(c.denominator, -1, P) % P

if __name__ == "__main__":
    for m in map(int, sys.argv[1:]):
        po = Po(m)
        reps, _ = type_reps(po)
        for ty, msk in sorted(reps.items()):
            x = {msk: 1}
            if not homologic(po, x): continue
            q = Quot(po, x)
            if q.dim == 0: continue
            T = np.array([[[modp(c) for c in v] for v in row] for row in q.table()], dtype=np.int64)
            s, steps = radical_quotient(Alg(T, q.par))
            o = f"m={m} {po.name(msk):>22} rank={q.rank:>3} sdim g_x={q.sdim()} r-steps={steps} sdim g_x/r={s.sdim()}"
            if s.d:
                o += f" dim s'={s.derived_dim()} centroid<={s.centroid_bound()}"
                if s.par.count(1) == 0:
                    o += f" toral rank={s.toral_rank()}"
            print(o, flush=True)
