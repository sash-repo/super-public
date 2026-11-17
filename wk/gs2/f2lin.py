"""F_2 linear algebra on numpy uint8 matrices (rows)."""
import numpy as np
def rref(M):
    M = (np.array(M, dtype=np.uint8) & 1).copy()
    r = 0; piv = []
    rows, cols = M.shape
    for c in range(cols):
        if r >= rows: break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0: continue
        p = r + nz[0]
        if p != r: M[[r, p]] = M[[p, r]]
        mask = M[:, c].astype(bool); mask[r] = False
        M[mask] ^= M[r]
        piv.append(c); r += 1
    return M[:r], piv
def rank(M):
    if len(M) == 0: return 0
    return len(rref(M)[1])
def nullspace(M, ncols=None):
    M = np.array(M, dtype=np.uint8)
    if M.size == 0:
        return np.eye(ncols, dtype=np.uint8)
    R, piv = rref(M); n = M.shape[1]
    free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8); v[f] = 1
        for i, pc in enumerate(piv): v[pc] = R[i, f]
        out.append(v)
    return np.array(out, dtype=np.uint8).reshape(len(out), n)
def solve(M, b):
    """a solution x of M x = b or None"""
    M = np.array(M, dtype=np.uint8); b = np.array(b, dtype=np.uint8).reshape(-1, 1)
    R, piv = rref(np.hstack([M, b]))
    n = M.shape[1]
    if n in piv: return None
    x = np.zeros(n, dtype=np.uint8)
    for i, pc in enumerate(piv): x[pc] = R[i, n]
    return x

class Solver:
    """fast repeated solves of M x = b over F_2: x supported on pivot columns, checked afterwards"""
    def __init__(s, M):
        s.M = np.array(M, dtype=np.uint8)
        R, piv = rref(s.M.T)          # row-pivots of M = pivot columns of M^T -> independent rows of M
        rows = piv
        R2, piv2 = rref(s.M[rows])      # independent columns among those rows
        cols = piv2
        A = s.M[np.ix_(rows, cols)]
        n = len(rows)
        aug = np.hstack([A, np.eye(n, dtype=np.uint8)])
        Ra, pa = rref(aug)
        assert pa[:n] == list(range(n))
        s.Ainv = Ra[:, n:].astype(np.int64); s.rows = rows; s.cols = cols
    def solve(s, b):
        b = np.asarray(b, dtype=np.int64)
        xc = (s.Ainv @ b[s.rows]) & 1
        x = np.zeros(s.M.shape[1], dtype=np.int64); x[s.cols] = xc
        if (((s.M.astype(np.int64) @ x) & 1) != (b & 1)).any(): return None
        return x.astype(np.uint8)
