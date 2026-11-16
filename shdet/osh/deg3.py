import sympy as sp, sys
from osh import *
lam = sp.Rational(int(sys.argv[1]), int(sys.argv[2])); t = sp.Rational(int(sys.argv[3]), int(sys.argv[4]))
deg = int(sys.argv[5]); Ns = [int(a) for a in sys.argv[6:]]
Ua = U(lam, 'QQ'); Ka = K(Ua)
V = Verma(Ka, lambda j: t**(2*j+1))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "lib"))
from exactlin import rank
for N in Ns:
    ms, G = gram(V, deg, N)
    print(f"lam={lam} t={t} deg -{deg} N={N} size={len(ms)} rank={rank([[sp.Rational(x) for x in G.row(i)] for i in range(G.rows)])}", flush=True)
