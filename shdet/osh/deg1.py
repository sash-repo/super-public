import sympy as sp
from osh import *
lam = sp.Symbol('lam')
Ua = U(lam, 'QQ[lam]'); Ka = K(Ua)
x = sp.Symbol('x')
def G(N, chi):
    return sp.Matrix(N, N, lambda i, k: sum(c.as_expr()*chi(j) if hasattr(c,'as_expr') else c*chi(j)
                                            for (d, j), c in Ka.brk((1, i), (-1, k))))
cs = sp.symbols('c0:12')
N = 5; GG = G(N, lambda j: cs[j]); ok = 0
def chi(poly):
    p = sp.Poly(sp.expand(poly), h); tot = 0
    for (k,), c in p.terms(): tot += c*cs[(k-1)//2]
    return sp.expand(tot)
for i in range(N):
    for k in range(N):
        n = i+k+1; g = (lam**2-x**2)*x**(2*n)/4
        assert sp.expand(GG[i,k] - chi(g.subs(x,h-1)-g.subs(x,h+1))) == 0; ok += 1
print("Hankel formula (F_i v|F_k v)=chi(g_n(h-1)-g_n(h+1)), n=i+k+1: ok on", ok, "entries")
t1,t2,t3,m1,m2,m3 = sp.symbols('t1 t2 t3 m1 m2 m3')
al = lambda t: (lam**2-(t-1)**2)*(t-1)**2/4
be = lambda t: (lam**2-(t+1)**2)*(t+1)**2/4
for ts, mm in [([t1],[1]), ([t1,t2],[m1,m2])]:
    r = len(ts)
    chi_ = lambda j: sum(m*t**(2*j+1) for m,t in zip(mm,ts))
    D = sp.factor(G(2*r, chi_).det())
    nodes, wts = [], []
    for m,t in zip(mm,ts): nodes += [(t-1)**2,(t+1)**2]; wts += [m*al(t), -m*be(t)]
    pred = sp.prod(wts)*sp.prod([(nodes[a]-nodes[b])**2 for a in range(2*r) for b in range(a+1,2*r)])
    print("r =", r, "  det_{2r} =", D)
    print("     = prod(59)*Vandermonde^2:", sp.expand(D-pred) == 0)
    print("     rank of the", 2*r+2, "-truncation (generic numeric point):",
          G(2*r+2, chi_).subs({lam:sp.Rational(7,3), t1:sp.Rational(2,5), t2:sp.Rational(9,4), m1:3, m2:sp.Rational(-5,2)}).rank())
