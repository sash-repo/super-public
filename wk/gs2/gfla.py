"""GF(2^K) linear algebra on numpy int arrays (K=1 is F_2).  Elements are ints 0..2^K-1."""
import numpy as np
MODS = {1: 0b11, 2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011, 8: 0b100011011}

class GF:
    def __init__(s, K):
        s.K = K; s.q = 1 << K; m = MODS[K]
        def mulraw(a, b):
            r = 0
            while b:
                if b & 1: r ^= a
                b >>= 1; a <<= 1
                if a >> K: a ^= m
            return r
        s.mulraw = mulraw
        if K == 1:
            g = 1
        else:
            for g in range(2, s.q):
                x, seen = 1, set()
                for _ in range(s.q - 1):
                    seen.add(x); x = mulraw(x, g)
                if len(seen) == s.q - 1: break
        exp = np.zeros(2 * s.q, dtype=np.int64); x = 1
        for i in range(s.q - 1):
            exp[i] = x; x = mulraw(x, g)
        for i in range(s.q - 1, 2 * s.q): exp[i] = exp[i - (s.q - 1)]
        log = np.zeros(s.q, dtype=np.int64)
        for i in range(s.q - 1): log[exp[i]] = i
        s.exp, s.log = exp, log
        # full multiplication table for small q
        a = np.arange(s.q)
        s.T = np.array([[mulraw(int(i), int(j)) for j in a] for i in a], dtype=np.int64)
        s.INV = np.zeros(s.q, dtype=np.int64)
        for i in range(1, s.q): s.INV[i] = exp[(s.q - 1 - log[i]) % (s.q - 1)]
    def mul(s, a, b):
        return s.T[a, b]
    def sq(s, a):
        return s.T[a, a]
    def sqrt(s, a):          # Frobenius inverse
        r = a
        for _ in range(s.K - 1): r = s.T[r, r]
        return r
    def dot(s, A, B):
        """matrix product A (m x n) . B (n x p) over GF"""
        A = np.asarray(A); B = np.asarray(B)
        out = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
        for k in range(A.shape[1]):
            out ^= s.T[A[:, k][:, None], B[k, :][None, :]]
        return out
    def rref(s, M):
        M = np.array(M, dtype=np.int64).copy()
        if M.size == 0: return M, []
        rows, cols = M.shape; r = 0; piv = []
        for c in range(cols):
            if r >= rows: break
            nz = np.nonzero(M[r:, c])[0]
            if len(nz) == 0: continue
            p = r + nz[0]
            if p != r: M[[r, p]] = M[[p, r]]
            M[r] = s.T[s.INV[M[r, c]], M[r]]
            col = M[:, c].copy(); col[r] = 0
            nzr = np.nonzero(col)[0]
            if len(nzr):
                M[nzr] ^= s.T[col[nzr][:, None], M[r][None, :]]
            piv.append(c); r += 1
        return M[:r], piv
    def rank(s, M):
        M = np.asarray(M)
        if M.size == 0: return 0
        return len(s.rref(M)[1])
    def nullspace(s, M, n=None):
        M = np.asarray(M, dtype=np.int64)
        if M.size == 0: return np.eye(n, dtype=np.int64)
        R, piv = s.rref(M); n = M.shape[1]
        free = [c for c in range(n) if c not in piv]
        out = np.zeros((len(free), n), dtype=np.int64)
        for k, f in enumerate(free):
            out[k, f] = 1
            for i, pc in enumerate(piv): out[k, pc] = R[i, f]   # char 2: -R = R
        return out
