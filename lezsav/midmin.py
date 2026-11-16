"""midmin.py -- test of the middle rows: minimize the norm
of the four middle rows n_z in {-2,-1,0,1} ALONE over all of lam (other rows
ignored), and, separately, the full |Phi| with only lam_0, lam_1, lam_2 free.
Phi is exactly quadratic, so the Jacobian is J(lam) = L + 2B(lam,.), taken by
central differences (exact up to rounding)."""
import sys, os, pickle, numpy as np, collections
TAG, MODE, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.argv = [sys.argv[0], TAG]
exec(open("peel.py").read().split("blocks = sorted")[0].split('TAG = sys.argv[1]')[1])
MID = (-2, -1, 0, 1)
if MODE == "mid":
    free = np.arange(N)
    def fun(V):
        L = np.zeros((N, V.shape[1]), complex); L[free] = V
        R = lm2.resid(L)
        return np.vstack([rows(R, z) for z in MID])
else:
    free = np.nonzero((KZ >= 0) & (KZ <= 2))[0]
    def fun(V):
        L = np.zeros((N, V.shape[1]), complex); L[free] = V
        return lm2.resid(L)
n = len(free)
rng = np.random.default_rng(seed)
x = np.zeros(n, complex) if seed == 0 else 0.3*(rng.standard_normal(n) + 1j*rng.standard_normal(n))
r = fun(x[:, None])[:, 0]; f = np.linalg.norm(r); mu = 1e-3
print("%s seed %d: %d unknowns, %d components, start %.5g" % (MODE, seed, n, len(r), f), flush=True)
for it in range(60):
    J = jacF(fun, x, n)
    A = J.conj().T @ J; g = J.conj().T @ r
    while True:
        d = np.linalg.solve(A + mu*np.diag(np.diag(A).real + 1e-12), -g)
        r2 = fun((x + d)[:, None])[:, 0]; f2 = np.linalg.norm(r2)
        if f2 < f: x, r, f = x + d, r2, f2; mu = max(mu/3, 1e-12); break
        mu *= 10
        if mu > 1e12: break
    if it % 5 == 0 or mu > 1e12: print("  it %d  %.6g  |x| %.3g" % (it, f, np.linalg.norm(x)), flush=True)
    if mu > 1e12 or f < 1e-10: break
print("%s seed %d FINAL %.6g |x| %.3g" % (MODE, seed, f, np.linalg.norm(x)), flush=True)
pickle.dump(dict(x=x, f=f, free=free), open("midmin-%s-%s-%d.pkl" % (TAG, MODE, seed), "wb"))
