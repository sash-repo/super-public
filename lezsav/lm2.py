"""lm2.py -- Levenberg-Marquardt search for zeros of Phi on the
448-dimensional kernel of the level-0 system (loop F = A zeta, level-2 window
[-3,4], level-4 window [-10,7]).  Float evaluation by lanes.py; Jacobian by
the holomorphic 4-point stencil (exact for degree <= 4 up to rounding).
Usage: lm2.py TAG <seed> <scale> <iters>   (seed 0 = start at the particular solution)"""
import sys, os, time, random, pickle
import numpy as np
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import lanes
import sys as _s
TAG = _s.argv[1]
E = lanes.Engine2(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build-%s.pkl" % TAG))
N = E.nker
def resid(L):
    Y, F = E.phi(L)
    if not hasattr(E, "rown"):
        E.rown = np.ones(E.Ymat.shape[0])
        E.fkeys = None
    Y = Y * E.rown[:, None]
    if E.fkeys is None:
        rr = np.random.default_rng(7)
        Fr = E.phi(rr.standard_normal((N, 3)) + 1j * rr.standard_normal((N, 3)))[1]
        E.fkeys = sorted(set(Fr) | set(F), key=str)
        print("free rows:", len(E.fkeys), flush=True)
    Fm = np.array([F.get(k, np.zeros(L.shape[1])) for k in E.fkeys]).reshape(len(E.fkeys), L.shape[1])
    for k in F:
        if k not in E.fkeys and np.any(np.abs(F[k]) > 1e-9):
            print("WARNING new free key", k)
    return np.vstack([Y, Fm])
def jac(lam, chunk=48):
    cols = []
    for s in range(0, N, chunk):
        idx = range(s, min(N, s + chunk))
        Ls = []
        for j in idx:
            for t in (-2, -1, 1, 2):
                v = lam.copy(); v[j] += t; Ls.append(v)
        R = resid(np.array(Ls).T)
        for n, j in enumerate(idx):
            f = R[:, 4 * n:4 * n + 4]
            cols.append((8 * (f[:, 2] - f[:, 1]) - (f[:, 3] - f[:, 0])) / 12)
    return np.array(cols).T
if __name__ == "__main__":
    seed, scale, iters = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    rng = np.random.default_rng(seed)
    lam = np.zeros(N, complex) if seed == 0 else scale * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    r = resid(lam[:, None])[:, 0]; f = np.linalg.norm(r); mu = 1e-3
    print("seed %d start |Phi| = %.6g (%d components)" % (seed, f, len(r)), flush=True)
    hist = []
    for it in range(iters):
        t0 = time.time()
        J = jac(lam)
        g = J.conj().T @ r
        A = J.conj().T @ J
        sv = np.linalg.svd(J, compute_uv=False)
        while True:
            d = np.linalg.solve(A + mu * np.diag(np.diag(A).real + 1e-12), -g)
            r2 = resid((lam + d)[:, None])[:, 0]; f2 = np.linalg.norm(r2)
            if f2 < f:
                lam, r, f = lam + d, r2, f2; mu = max(mu / 3, 1e-12); break
            mu *= 10
            if mu > 1e12: break
        hist.append(f)
        print("it %d |Phi| = %.6g  mu %.1e  rank(J) %d/%d  smin %.3g  |grad| %.3g [%.0fs]"
              % (it, f, mu, int(np.sum(sv > sv[0] * 1e-9)), N, sv[-1], np.linalg.norm(g), time.time() - t0), flush=True)
        if mu > 1e12 or f < 1e-9: break
    pickle.dump(dict(lam=lam, r=r, hist=hist), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm2-%s-%d.pkl" % (TAG, seed)), "wb"))
    big = np.argsort(-np.abs(r))[:10]
    print("final |Phi| = %.6g; largest components %s" % (f, np.round(np.abs(r[big]), 4)))
