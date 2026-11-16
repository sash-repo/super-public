"""peel2.py -- structure of the edge quadrics Q_k(v)=Q(v,v) at
row 2k-2 (k<=-1) / 2k-1 (k>=3): the symmetric bilinear form B_k(u,v),
written as a matrix (rows x 56 x 56); the subspace S_k of v with B_k(v,.)=0
(the radical) and the rank of the map v -> B_k(v,.)."""
import sys, os, pickle, numpy as np, collections
TAG = sys.argv[1]
sys.argv = [sys.argv[0], TAG]
exec(open("peel.py").read().split("blocks = sorted")[0].split('TAG = sys.argv[1]')[1].replace("import lm2","import lm2",1))
blocks = sorted(set(KZ))
out = {}
for k in blocks:
    if -1 < k < 3: continue
    z = 2*k - 2 if k <= -1 else 2*k - 1
    m = np.nonzero(KZ == k)[0]; n = len(m)
    def fun(V):
        L = np.zeros((N, V.shape[1]), complex); L[m] = V
        return F(L, z)
    # bilinear form by polarization: B(e_i,e_j) = (Q(ei+ej)-Q(ei)-Q(ej))/2
    I = np.eye(n)
    cols = [I[i] for i in range(n)] + [I[i] + I[j] for i in range(n) for j in range(i+1, n)]
    R = fun(np.array(cols).T)
    q = R[:, :n]
    Bm = np.zeros((R.shape[0], n, n), complex)
    for i in range(n): Bm[:, i, i] = q[:, i]
    c = n
    for i in range(n):
        for j in range(i+1, n):
            Bm[:, i, j] = Bm[:, j, i] = (R[:, c] - q[:, i] - q[:, j]) / 2; c += 1
    # radical: v with Bm[:, :, :] v = 0  -> stack over rows and second index
    M = Bm.transpose(0, 2, 1).reshape(-1, n)   # (rows*n) x n
    s = np.linalg.svd(M, compute_uv=False)
    rk = int(np.sum(s > s[0]*1e-9))
    # check: random v in radical complement?  zero set dimension proxy: rank of J at random v
    v = np.random.default_rng(0).standard_normal(n) + 0j
    Jv = 2*np.einsum("rij,j->ri", Bm, v)
    sj = np.linalg.svd(Jv, compute_uv=False)
    print("k=%d row %d: %d eqs, n=%d, rank of v->B(v,.) = %d (radical dim %d), rank of J at random v = %d"
          % (k, z, R.shape[0], n, rk, n - rk, int(np.sum(sj > sj[0]*1e-9))), flush=True)
    out[k] = Bm
pickle.dump(out, open("peel2-%s.pkl" % TAG, "wb"))
