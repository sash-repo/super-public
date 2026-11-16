"""GF(4) = {0,1,w,w^2} encoded 0,1,2,3 (w^2 = w+1): numpy linear algebra and structure-constant
invariants of a Lie superalgebra (tab[i,j,k], sq[i,k], par)."""
import numpy as np, itertools
def _m(a, b):
    r = 0
    while b:
        if b & 1: r ^= a
        b >>= 1; a <<= 1
        if a & 4: a ^= 0b111
    return r
MUL = np.array([[_m(a, b) for b in range(4)] for a in range(4)], dtype=np.uint8)
INV = np.array([0, 1, 3, 2], dtype=np.uint8)
def mul(a, b): return MUL[a, b]
def rref(M):
    M = np.array(M, dtype=np.uint8).copy(); rows, cols = M.shape; r = 0; piv = []
    for c in range(cols):
        if r >= rows: break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0: continue
        p = r + nz[0]
        if p != r: M[[r, p]] = M[[p, r]]
        M[r] = MUL[INV[M[r, c]], M[r]]
        for i in np.nonzero(M[:, c])[0]:
            if i != r: M[i] ^= MUL[M[i, c], M[r]]
        piv.append(c); r += 1
    return M[:r], piv
def rank(M):
    M = np.asarray(M)
    return 0 if M.size == 0 else len(rref(M)[1])
def matmul(A, B):
    """A (n x m) * B (m x k) over GF(4)"""
    A = np.asarray(A, dtype=np.uint8); B = np.asarray(B, dtype=np.uint8)
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.uint8)
    for l in range(A.shape[1]):
        out ^= MUL[A[:, l][:, None], B[l][None, :]]
    return out
def nullspace(M, n):
    M = np.asarray(M, dtype=np.uint8)
    if M.size == 0: return np.eye(n, dtype=np.uint8)
    R, piv = rref(M); free = [c for c in range(n) if c not in piv]; out = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8); v[f] = 1
        for i, pc in enumerate(piv): v[pc] = R[i, f]   # char 2: -R = R
        out.append(v)
    return np.array(out, dtype=np.uint8).reshape(len(out), n)

class SA:
    def __init__(s, tab, sq, par):
        s.tab = np.asarray(tab, dtype=np.uint8); s.sq = np.asarray(sq, dtype=np.uint8); s.par = np.asarray(par); s.d = len(par)
    def br(s, x, y):
        d = s.d; out = np.zeros(d, dtype=np.uint8)
        for i in np.nonzero(x)[0]:
            for j in np.nonzero(y)[0]:
                out ^= MUL[MUL[x[i], y[j]], s.tab[i, j]]
        return out
    def square(s, x):
        out = np.zeros(s.d, dtype=np.uint8); idx = np.nonzero(x)[0]
        for i in idx: out ^= MUL[MUL[x[i], x[i]], s.sq[i]]
        for a, b in itertools.combinations(idx, 2): out ^= MUL[MUL[x[a], x[b]], s.tab[a, b]]
        return out
    def ad(s, i): return s.tab[i].T.copy()
    def derived(s):
        return rref(np.array([s.tab[i, j] for i in range(s.d) for j in range(s.d)]))[0]
    def sdim_of(s, B):
        return rank(np.array([v * (s.par == 0) for v in B])), rank(np.array([v * (s.par == 1) for v in B]))
    def center(s):
        d = s.d; M = s.tab.transpose(1, 2, 0).reshape(d * d, d)
        return nullspace(M, d)
    def assoc_dim(s):
        d = s.d; gens = [s.ad(i) for i in range(d)]
        B = rref(np.eye(d, dtype=np.uint8).reshape(1, -1))[0]; queue = [np.eye(d, dtype=np.uint8)]
        while queue:
            X = queue.pop()
            for M in gens:
                Y = matmul(M, X)
                t = rref(np.vstack([B, Y.reshape(1, -1)]))[0]
                if len(t) > len(B): B = t; queue.append(Y)
        return len(B)
    def ideal(s, gens):
        d = s.d; E = np.eye(d, dtype=np.uint8); B = rref(np.array(gens))[0]
        while True:
            new = [s.br(E[i], v) for v in B for i in range(d)]
            new += [s.square(v) for v in B if v.any() and (s.par[np.nonzero(v)[0]] == 1).all()]
            nb = rref(np.vstack([B] + [np.array(new)]))[0]
            if len(nb) == len(B): return B
            B = nb
    def restricted(s):
        ev = [i for i in range(s.d) if s.par[i] == 0]
        A = rref(np.array([s.ad(i).reshape(-1) for i in ev]))[0]; bad = 0
        for i in ev:
            M = s.ad(i); M2 = matmul(M, M)
            if rank(np.vstack([A, M2.reshape(1, -1)])) > len(A): bad += 1
        return bad == 0
    def odd_null_squares(s):
        """number of odd x != 0 with x^2 = 0 (over GF(4)), via the quadratic form coordinates"""
        idx = np.nonzero(s.par == 1)[0]; n = 0
        for c in itertools.product(range(4), repeat=len(idx)):
            if not any(c): continue
            x = np.zeros(s.d, dtype=np.uint8); x[idx] = c
            if not s.square(x).any(): n += 1
        return n
    def der_sdim(s):
        d = s.d; res = {}
        T = s.tab
        for pD in (0, 1):
            idx = [(l, k) for l in range(d) for k in range(d) if (s.par[l] + s.par[k]) % 2 == pD]
            cols = []
            for (l, k) in idx:
                # phi(e_l) = e_k
                E = np.zeros((d, d, d), dtype=np.uint8)
                E ^= (T[:, :, l][:, :, None] * (np.arange(d) == k)[None, None, :]).astype(np.uint8)   # phi([a,b])
                E[l, :, :] ^= T[k, :, :]                                                          # [phi a, b] for a=l
                E[:, l, :] ^= T[:, k, :]                                                          # [a, phi b] for b=l
                S = np.zeros((d, d), dtype=np.uint8)
                S ^= (s.sq[:, l][:, None] * (np.arange(d) == k)[None, :]).astype(np.uint8)
                S[l] ^= T[l, k]
                S = S[s.par == 1]
                cols.append(np.concatenate([E.reshape(-1), S.reshape(-1)]))
            M = np.array(cols, dtype=np.uint8).T
            res[pD] = len(idx) - rank(M)
        return res
    def is_simple_sample(s, tries=400, seed=0):
        """ideals generated by random homogeneous elements + all basis elements (a sufficient
        test for non-simplicity; simplicity is claimed only if every one-element ideal is everything)"""
        rng = np.random.default_rng(seed); d = s.d
        cand = [np.eye(d, dtype=np.uint8)[i] for i in range(d)]
        for _ in range(tries):
            p = rng.integers(2); v = rng.integers(0, 4, d).astype(np.uint8) * (s.par == p)
            if v.any(): cand.append(v.astype(np.uint8))
        for v in cand:
            if len(s.ideal([v])) < d: return False
        return True
