"""Deformations of a Lie superalgebra over F_2 with bracket and squaring (char 2).
A structure is (T, S): T[a,b,k] symmetric in a,b (char 2), T[a,a]=0, S[i,k] squaring of odd e_i.
Identities: Jacobi on distinct triples, [s(x),y] = [x,[x,y]] for odd basis x, all y.
Deformation mu_t = sum t^n (T_n, S_n); the t^n part of the identities is
sum_{j+k=n} R(T_j, T_k, S_k)."""
import numpy as np, itertools
from f2lin import rref, rank, nullspace, solve

class Deform:
    def __init__(s, tab, sq, par):
        s.T0 = tab.astype(np.int64); s.S0 = sq.astype(np.int64); s.par = par; d = s.d = len(par)
        s.odd = [i for i in range(d) if par[i]]
        s.trip = np.array(list(itertools.combinations(range(d), 3)))
        # cochain variables
        s.vars = []
        for a in range(d):
            for b in range(a + 1, d):
                for k in range(d):
                    if (par[a] + par[b] + par[k]) % 2 == 0: s.vars.append(('c', a, b, k))
        for a in s.odd:
            for k in range(d):
                if par[k] == 0: s.vars.append(('q', a, a, k))
        s.nv = len(s.vars)
        s.D = s.delta_matrix()
    def unpack(s, v):
        d = s.d; C = np.zeros((d, d, d), dtype=np.int64); Q = np.zeros((d, d), dtype=np.int64)
        for x, (t, a, b, k) in zip(v, s.vars):
            if x:
                if t == 'c': C[a, b, k] = C[b, a, k] = 1
                else: Q[a, k] = 1
        return C, Q
    def pack(s, C, Q):
        return np.array([(C[a, b, k] if t == 'c' else Q[a, k]) & 1 for (t, a, b, k) in s.vars], dtype=np.uint8)
    def R(s, Tj, Tk, Sk):
        X = np.einsum('alk,bcl->abck', Tj, Tk)
        a, b, c = s.trip[:, 0], s.trip[:, 1], s.trip[:, 2]
        J = X[a, b, c] + X[b, c, a] + X[c, a, b]
        o = s.odd
        Y = np.einsum('il,lyk->iyk', Sk[o], Tj) + np.einsum('ilk,iyl->iyk', Tj[o], Tk[o])
        return np.concatenate([J.reshape(-1), Y.reshape(-1)]) & 1
    def lin(s, C, Q):
        return (s.R(s.T0, C, Q) + s.R(C, s.T0, s.S0)) & 1
    def delta_matrix(s):
        cols = []
        for i in range(s.nv):
            v = np.zeros(s.nv, dtype=np.uint8); v[i] = 1
            cols.append(s.lin(*s.unpack(v)))
        return np.array(cols, dtype=np.uint8).T      # equations x vars
    def coboundaries(s):
        d = s.d; out = []
        for l in range(d):
            for k in range(d):
                if s.par[l] != s.par[k]: continue
                P = np.zeros((d, d), dtype=np.int64); P[l, k] = 1     # phi(e_l) = e_k ; P[l] row = image
                C = np.einsum('abl,lk->abk', s.T0, P) + np.einsum('al,lbk->abk', P, s.T0) + np.einsum('bl,alk->abk', P, s.T0)
                Q = s.S0 @ P + np.einsum('al,alk->ak', P, s.T0)
                for a in range(d): C[a, a] = 0
                out.append(s.pack(C & 1, Q & 1))
        return np.array(out, dtype=np.uint8)
    def cohomology(s):
        Z = nullspace(s.D, s.nv)
        B = s.coboundaries()
        rB = rank(B)
        assert rank(np.vstack([Z, B])) == len(Z), 'coboundary not a cocycle'
        # complement of B in Z
        Rb, _ = rref(B)
        reps = []
        cur = Rb.copy()
        for z in Z:
            t = np.vstack([cur, z])
            if rank(t) > len(cur) if len(cur) else z.any():
                reps.append(z); cur = rref(t)[0]
        s.Z, s.B, s.H = Z, Rb, np.array(reps, dtype=np.uint8)
        return len(Z), rB, len(reps)
