"""cert.py -- look for a degree-0 certificate: vectors y with
y^H J(lam) = 0 at every lam (so y^H Phi is constant).  Then
|Phi(lam)| >= |P_W Phi(0)| for all lam, W = span of such y.
Usage: cert.py TAG npoints"""
import sys, os, time, pickle
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1]] + sys.argv[2:]
import lm2
E, N = lm2.E, lm2.N
m = int(sys.argv[2])
rng = np.random.default_rng(11)
Js = []
for k in range(m):
    lam = 0.7 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    t = time.time(); Js.append(lm2.jac(lam)); print("jac %d [%.0fs]" % (k, time.time() - t), flush=True)
    A = np.hstack(Js)
    U, s, _ = np.linalg.svd(A, full_matrices=True)
    r = int(np.sum(s > s[0] * 1e-9))
    print("  stacked rank %d of %d components -> common left null dim %d" % (r, A.shape[0], A.shape[0] - r), flush=True)
W = U[:, r:]
r0 = lm2.resid(np.zeros((N, 1)))[:, 0]
c0 = W.conj().T @ r0
print("|P_W Phi(0)| = %.6f" % np.linalg.norm(c0))
for k in range(3):
    lam = 2.0 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    rl = lm2.resid(lam[:, None])[:, 0]
    print("  check: |W^H(Phi(lam)-Phi(0))| = %.3g   |Phi(lam)| = %.3f" % (np.linalg.norm(W.conj().T @ rl - c0), np.linalg.norm(rl)))
pickle.dump(dict(W=W, c0=c0, s=s, r=r), open("cert-%s.pkl" % sys.argv[1], "wb"))
