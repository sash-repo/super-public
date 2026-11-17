"""L = restricted closure of gs(2) (from shenL.npz) over GF(2^K): bracket, p-map, forms."""
import numpy as np
from gfla import GF
D = np.load('shenL.npz')
CL, SQ, WTS, NAMES, NG = D['CL'].astype(np.int64), D['SQ'].astype(np.int64), D['wts'], list(D['names']), int(D['ng'])
N = CL.shape[0]
def bracket(F, x, y):
    out = np.zeros(N, dtype=np.int64)
    for i in np.nonzero(x)[0]:
        for j in np.nonzero(y)[0]:
            c = F.T[x[i], y[j]]
            if c: out ^= F.T[c, CL[i, j]]
    return out
def psq(F, x):
    """x^[2] = sum x_i^2 e_i^[2] + sum_{i<j} x_i x_j [e_i,e_j]"""
    out = np.zeros(N, dtype=np.int64)
    nz = np.nonzero(x)[0]
    for a, i in enumerate(nz):
        out ^= F.T[F.T[x[i], x[i]], SQ[i]]
        for j in nz[a + 1:]:
            out ^= F.T[F.T[x[i], x[j]], CL[i, j]]
    return out
def admat(F, x):
    return np.array([bracket(F, x, np.eye(N, dtype=np.int64)[j]) for j in range(N)]).T
def Bchi(F, chi):
    """B[i,j] = chi([e_i,e_j])"""
    B = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            v = 0
            for k in np.nonzero(CL[i, j])[0]: v ^= chi[k]
            B[i, j] = v
    return B
