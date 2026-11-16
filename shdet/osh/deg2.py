"""dim of L^chi_{-d} for chi = evaluation at t (numeric lam,t), via ranks of truncated Gram matrices"""
import sympy as sp, sys
from osh import *
lam = sp.Rational(int(sys.argv[1]), int(sys.argv[2])); t = sp.Rational(int(sys.argv[3]), int(sys.argv[4]))
Ua = U(lam, 'QQ'); Ka = K(Ua)
V = Verma(Ka, lambda j: t**(2*j+1))
for deg in (1, 2, 3):
    for N in range(1, 6 if deg < 3 else 4):
        ms, G = gram(V, deg, N)
        print(f"lam={lam} t={t} deg -{deg} N={N} size={len(ms)} rank={G.rank()}", flush=True)
