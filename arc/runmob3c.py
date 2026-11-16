"""The same verdict as runmob3.py on sparse generating functions: cheap enough
to finish, and the twist is a first-order effect, so a truncation at the mixed
term sees it."""
import sys, sympy as sp
import radul as R, mobius as M
from radul import t, e, d, num, th, eadd, esub, emul, escal, eeq, einv, epow
from mobius import Diffeo, Composite, multiplier
from runmob3 import S3_M, twisted, check

N = 3
M.setup(N); R.set_order(2)
# 1 = zeta, 2 = theta_1, 3 = theta_2;  odd components carry odd constants
g = eadd({(): t**2}, escal(sp.Rational(1,2), {(1,2): t}), {(2,3): sp.Integer(1)},
         emul({(101,): sp.Integer(1)}, {(1,): t}))
h = eadd({(): t**3 + t}, {(1,3): t}, emul({(102,): sp.Integer(1)}, {(2,): t**2}))
G = Diffeo(g, e, N); H = Diffeo(h, d, N)
S = lambda X: S3_M(X, N, -1)
check("S_3 is not identically zero", bool(S(G)))
check("S_3(GH) = S_3(H) + m_H^{1/2}(H^*t/t)^{1/2} H^*(S_3(G))",
      twisted(S, sp.Rational(1, 2), N, G, H, 1))
for p in (0, -1, 2, 3):
    check("   the Moebius power %d/2 fails" % p,
          not twisted(S, sp.Rational(1, 2), N, G, H, p))
for w in (sp.Rational(3, 2), sp.Rational(-1, 2), sp.Integer(1)):
    check("   the weight %s fails" % w, not twisted(S, w, N, G, H, 1))
