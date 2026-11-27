"""Statement 5.A.8.2, the law (A.22): full check of the law found by oddlaw.py psi2:  beta'-beta = (-1)^{p(a)}(-Hess+dphi dphi - lam*u*g), i.e. rho'^0-rho^0 = -Hess+dphi dphi - lam*kappa*g
(beta = coordinate matrix with nabla_a d_s = (-1)^{p(a)+1} beta_as p + ...; rho^0_as = (-1)^{p(a)} beta_as)."""
import pickle, sympy as sp
from oddexample import A, N, P, x, y, metric, ev, evM
d = pickle.load(open('oddlaw.pkl', 'rb')); Q, H, dd = d['Q'], d['Hess'], d['dd']
lam = ev({(): 1 + x**2 + y, (0, 1): 1 + x}); g = evM(metric())
u = {(0,): sp.Symbol('u0'), (1,): sp.Symbol('u1')}
bad = 0
for a in range(N):
    for s in range(N):
        pred = A.sc((-1)**P(a), A.add(A.neg(H[a][s]), dd[a][s], A.neg(A.mul(A.mul(lam, u), g[a][s]))))
        r = A.clean({k: sp.expand(v) for k, v in A.add(Q[a][s], A.neg(pred)).items()})
        if r: bad += 1; print(a, s, r)
print("entries checked:", N*N, "mismatches:", bad)
