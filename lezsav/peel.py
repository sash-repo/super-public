"""peel.py -- the edge quadrics.  For k<=-1 the row n_z=2k-2
holds only Q(lam_k,lam_k); for k>=3 the row 2k-1 does.  Numerical test that
these homogeneous quadratic systems have no nonzero zero: minimize |Q_edge(v)|
on |v|=1 (LM on the sphere, many starts).  Then the reduced system in
lam_0, lam_1, lam_2: minimum of |Phi| with all other blocks zero."""
import sys, os, pickle, numpy as np, collections
TAG = sys.argv[1]
sys.argv = [sys.argv[0], TAG]
import lm2
E, N = lm2.E, lm2.N
ny = E.Ymat.shape[0]
zd = collections.defaultdict(set)
for (i, j, k), lst in E.ukeys.items():
    for d, _ in lst: zd[d].add(k[0])
KZ = np.array([next(iter(set().union(*[zd[d] for d in np.nonzero(np.abs(c) > 1e-12)[0]]))) for c in E.K.T])
zrow = np.zeros(len(E.yidx), int)
for k, n in E.yidx.items(): zrow[n] = k[2][0]
r0 = lm2.resid(np.zeros((N, 1)))
fz = np.array([k[2][0] for k in E.fkeys])
def rows(R, z):
    rv = E.Ymat.conj().T @ R[:ny]
    return np.vstack([rv[zrow == z], R[ny:][fz == z]])
def F(lam, z):   # residual rows at n_z = z, for lanes lam (N x L)
    return rows(lm2.resid(lam), z)
def jacF(fun, x, n):
    cols = []
    for s in range(0, n, 32):
        idx = list(range(s, min(n, s + 32))); Ls = []
        for j in idx:
            for t in (-1e-3, 1e-3):
                v = x.copy(); v[j] += t; Ls.append(v)
        R = fun(np.array(Ls).T)
        for m, j in enumerate(idx):
            cols.append((R[:, 2*m+1] - R[:, 2*m]) / 2e-3)
    return np.array(cols).T
blocks = sorted(set(KZ))
rng = np.random.default_rng(1)
res = {}
for k in blocks:
    if -1 < k < 3: continue
    z = 2*k - 2 if k <= -1 else 2*k - 1
    m = np.nonzero(KZ == k)[0]
    def fun(V):
        L = np.zeros((N, V.shape[1]), complex); L[m] = V
        return F(L, z)
    neq = fun(np.zeros((len(m), 1))).shape[0]
    best = []
    for st in range(4):
        v = rng.standard_normal(len(m)) + 1j*rng.standard_normal(len(m)); v /= np.linalg.norm(v)
        for it in range(40):
            r = fun(v[:, None])[:, 0]
            J = jacF(fun, v, len(m))
            # project out radial direction (homogeneous: J v = 2 r)
            P = np.eye(len(m)) - np.outer(v, v.conj())
            Jt = J @ P
            d = np.linalg.lstsq(Jt, -r, rcond=None)[0]
            v2 = v + 0.5*P@d; v2 /= np.linalg.norm(v2)
            if np.linalg.norm(fun(v2[:, None])) < np.linalg.norm(r): v = v2
            else: break
        best.append(np.linalg.norm(fun(v[:, None])))
    # also: smallest singular value of the 'Jacobian at v' family ~ generic
    print("k=%d row %d: %d equations in %d unknowns; min |Q(v)| on |v|=1 from 4 starts: %s"
          % (k, z, neq, len(m), " ".join("%.4g" % b for b in best)), flush=True)
