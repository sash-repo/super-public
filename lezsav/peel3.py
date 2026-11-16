"""peel3.py -- the edge quadric modulo its radical: min of
|Q(w)| over unit w orthogonal to the radical (LM on the sphere, 20 starts),
from the bilinear forms saved by peel2.py.  Pure numpy."""
import sys, pickle, numpy as np
D = pickle.load(open("peel2-%s.pkl" % sys.argv[1], "rb"))
rng = np.random.default_rng(2)
for k, Bm in sorted(D.items()):
    n = Bm.shape[1]
    M = Bm.transpose(0, 2, 1).reshape(-1, n)
    U, s, Vh = np.linalg.svd(M, full_matrices=True)
    rk = int(np.sum(s > s[0]*1e-9))
    C = Vh[:rk].conj().T            # n x rk, complement of radical (ON basis)
    Bc = np.einsum("ai,rab,bj->rij", C, Bm, C)   # holomorphic: use C (not conj) both sides? B is bilinear
    Bc = np.einsum("ai,rab,bj->rij", C, Bm, C)
    Q = lambda w: np.einsum("rij,i,j->r", Bc, w, w)
    best = []
    for st in range(20):
        w = rng.standard_normal(rk) + 1j*rng.standard_normal(rk); w /= np.linalg.norm(w)
        mu = 1e-2
        for it in range(300):
            r = Q(w); J = 2*np.einsum("rij,j->ri", Bc, w)
            P = np.eye(rk) - np.outer(w, w.conj())
            A = J @ P
            d = np.linalg.solve(A.conj().T@A + mu*np.eye(rk), -A.conj().T@r)
            w2 = w + P@d; w2 /= np.linalg.norm(w2)
            if np.linalg.norm(Q(w2)) < np.linalg.norm(r): w = w2; mu = max(mu/3, 1e-12)
            else: mu *= 10
            if mu > 1e8: break
        best.append(np.linalg.norm(Q(w)))
    sv = np.linalg.svd(Bc.reshape(Bc.shape[0], -1), compute_uv=False)
    print("k=%d: quotient rank %d; min |Q(w)| on |w|=1, w in complement: %.4g (max over starts %.4g); |B| scale %.3g"
          % (k, rk, min(best), max(best), sv[0]), flush=True)
