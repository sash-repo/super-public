"""struct.py -- polynomial structure of Phi on the 1010-dim
span V: degree along random lines, Phi(0)=c, L=J(0), the part of c and of
the quadratic part transverse to the image of L."""
import sys, os, pickle, numpy as np
sys.argv = [sys.argv[0], sys.argv[1]]
import lm2
E, N = lm2.E, lm2.N
C = pickle.load(open("cert-%s.pkl" % sys.argv[1], "rb"))
rng = np.random.default_rng(3)
r0 = lm2.resid(np.zeros((N, 1)))[:, 0]
# degree along lines: Phi(t v) for t = 0..6, finite differences
for k in range(2):
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    T = np.arange(7)
    R = lm2.resid(np.outer(v, T))
    # fit polynomial coefficients
    Vm = np.vander(T, 7, increasing=True)
    co = np.linalg.solve(Vm, R.T)   # 7 x comps
    print("line %d: |coef_k| =" % k, " ".join("%.3g" % np.linalg.norm(co[j]) for j in range(7)), flush=True)
J0 = lm2.jac(np.zeros(N, complex))
U, s, Vh = np.linalg.svd(J0, full_matrices=False)
rL = int(np.sum(s > s[0] * 1e-9))
print("rank L = rank J(0) = %d; |Phi(0)| = %.4f" % (rL, np.linalg.norm(r0)))
Ul = U[:, :rL]
c_perp = r0 - Ul @ (Ul.conj().T @ r0)
print("|c transverse to im L| = %.4f" % np.linalg.norm(c_perp))
pickle.dump(dict(r0=r0, J0=J0, rL=rL, U=U, s=s, Vh=Vh), open("phistruct-%s.pkl" % sys.argv[1], "wb"))
