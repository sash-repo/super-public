"""(1) the closed form J_k satisfies the chain equations al_{i+1}(a)c_{i+1,j}+(-1)^i al_{j+1}(b)c_{i,j+1}=0 identically, k<=20;
(2) J_2 at a=b=-2 vs the contact bracket [K_f,K_g]=K_{h}  (Corollary 7.6, Proposition 7.8).
Usage: python3 binJk.py"""
import sympy as sp
from binC import J, a, b
def al(n, x): return sp.Rational(n, 2) if n % 2 == 0 else x + (n-1)//2
for k in range(21):
    cs = J(k)
    for i in range(k):
        j = k-1-i
        e = al(i+1, a)*cs.get((i+1, j), 0) + (-1)**i*al(j+1, b)*cs.get((i, j+1), 0)
        assert sp.expand(e) == 0, (k, i)
print("closed form J_k satisfies the chain equations for k<=20")
t = sp.symbols('t')
f0, f1, g0, g1, u0, u1 = [sp.Function(n)(t) for n in ('f0', 'f1', 'g0', 'g1', 'u0', 'u1')]
# K_f on h=(h0,h1) (h=h0+h1 theta), f = f0 + f1 theta with f1 odd coefficient handled via parity:
# work with even f = f0 (K = 2 f0 d_t + f0' E) and odd f = f1 theta (K = f1 (theta d_t - d_theta)).
def Kev(f, h): return (2*f*sp.diff(h[0], t), 2*f*sp.diff(h[1], t) + sp.diff(f, t)*h[1])
def Kod(f, h): return (-f*h[1], f*sp.diff(h[0], t))
def J2(F, pF, G, a_, b_):   # F,G pairs; J_2 = b d(F)G + (-1)^{pF} D F D G - a F d(G)
    D = lambda h: (h[1], sp.diff(h[0], t)); d = lambda h: (sp.diff(h[0], t), sp.diff(h[1], t))
    def m(x, y): return (x[0]*y[0], x[0]*y[1] + x[1]*y[0])
    A, B_, C = m(d(F), G), m(D(F), D(G)), m(F, d(G))
    return tuple(sp.expand(b_*A[i] + (-1)**pF*B_[i] - a_*C[i]) for i in range(2))
h = (u0, u1)
# even-even: [K_{f0},K_{g0}] h
br = tuple(sp.expand(x - y) for x, y in zip(Kev(f0, Kev(g0, h)), Kev(g0, Kev(f0, h))))
j = J2((f0, 0), 0, (g0, 0), -2, -2)
tgt = Kev(j[0], h)
print("even-even: [K_f,K_g] = K_{c J_2(f,g)}, c =", sp.simplify(br[0]/tgt[0]) if tgt[0] != 0 else None)
