"""Induced modules u(L,chi) (x)_{u(P,chi)} K_lambda over GF(2^K), P spanned by basis vectors of L.
Basis of the module: subsets of Y (ordered complement), vectors = numpy arrays of length 2^|Y|."""
import numpy as np, sys
from functools import lru_cache
from lalg import CL, SQ, N
sys.setrecursionlimit(100000)

class Induced:
    def __init__(s, F, P, lam, chi):
        s.F = F; s.P = list(P); s.Y = [k for k in range(N) if k not in s.P]
        s.m = len(s.Y); s.dim = 1 << s.m; s.pos = {k: t for t, k in enumerate(s.Y)}
        s.lam = {k: lam[i] for i, k in enumerate(s.P)}; s.chi = np.asarray(chi)
        T = F.T
        # consistency: P closed, lambda a chi-character
        for a in s.P:
            for b in s.P:
                v = CL[a, b]
                assert not any(v[k] for k in s.Y), 'P not closed'
                assert s.lamv(v) == 0, 'lambda([P,P]) != 0'
            v = SQ[a]; assert not any(v[k] for k in s.Y), 'P not p-closed'
            assert T[s.lam[a], s.lam[a]] ^ s.lamv(v) == T[s.chi[a], s.chi[a]], 'lambda^2-lambda(x^[2]) != chi^2'
        s.cache = {}
    def lamv(s, v):
        out = 0
        for k in np.nonzero(v)[0]: out ^= s.lam[k]
        return out
    def mono(s, mask, c=1):
        v = np.zeros(s.dim, dtype=np.int64); v[mask] = c; return v
    def actv(s, x, mask):
        out = np.zeros(s.dim, dtype=np.int64)
        for j in np.nonzero(x)[0]:
            out ^= s.act(j, mask) if x[j] == 1 else s.F.T[x[j], s.act(j, mask)]
        return out
    def actY(s, t, vec):
        out = np.zeros(s.dim, dtype=np.int64); k = s.Y[t]
        for mm in np.nonzero(vec)[0]:
            c = vec[mm]; a = s.act(k, int(mm))
            out ^= a if c == 1 else s.F.T[c, a]
        return out
    def act(s, k, mask):
        key = (k, mask)
        r = s.cache.get(key)
        if r is not None: return r
        T = s.F.T
        if mask == 0:
            r = s.mono(0, s.lam[k]) if k in s.lam else s.mono(1 << s.pos[k])
        else:
            t0 = (mask & -mask).bit_length() - 1; rest = mask ^ (1 << t0)
            if k in s.pos:
                t = s.pos[k]
                if t < t0:
                    r = s.mono(mask | (1 << t))
                elif t == t0:
                    c = T[s.chi[k], s.chi[k]]
                    r = s.actv(SQ[k], rest) ^ s.mono(rest, c)
                else:
                    r = s.actY(t0, s.act(k, rest)) ^ s.actv(CL[k, s.Y[t0]], rest)
            else:
                r = s.actY(t0, s.act(k, rest)) ^ s.actv(CL[k, s.Y[t0]], rest)
        s.cache[key] = r
        return r
    def mat(s, k):
        return np.array([s.act(k, mm) for mm in range(s.dim)]).T      # column mm = e_k . basis_mm
    def mats(s):
        return [s.mat(k) for k in range(N)]

def check_rep(F, M, chi):
    """rho is a u(L,chi)-representation: brackets and p-map"""
    Id = np.eye(M[0].shape[0], dtype=np.int64)
    def comb(x):
        out = np.zeros_like(M[0])
        for j in np.nonzero(x)[0]: out ^= F.T[x[j], M[j]]
        return out
    for i in range(N):
        for j in range(N):
            lhs = F.dot(M[i], M[j]) ^ F.dot(M[j], M[i])
            assert (lhs == comb(CL[i, j])).all(), ('bracket', i, j)
        c = F.T[chi[i], chi[i]]
        assert ((F.dot(M[i], M[i]) ^ comb(SQ[i])) == F.T[c, Id]).all(), ('square', i)
    return True
