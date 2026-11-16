import sympy as sp, random
from osh import *
lam = sp.Rational(7,3)
Ua = U(lam, 'QQ'); Ka = K(Ua)
P = Ua.P
# relations: [e,f]=h via u, C value
e = {1: P(1)}; f = {-1: P(1)}; H = {0: P(h)}
assert Ua.br(e, f) == {0: P(h)}, Ua.br(e,f)
assert Ua.br(H, e) == {1: P(2)}
C = Ua.mul(e,f); C2 = Ua.mul(f,e); 
cas = {0: 2*C[0] + 2*C2[0] + P(h**2)}
assert cas == {0: P(lam**2-1)}, cas
# associativity spot check
X = {2: P(h**2+1), -1: P(h)}; Y = {-3: P(h-2), 1: P(3)}; Z = {1: P(h**3), -2: P(1)}
a = Ua.mul(Ua.mul(X,Y),Z); b = Ua.mul(X,Ua.mul(Y,Z))
assert all((a.get(d,P(0))-b.get(d,P(0))).is_zero for d in set(a)|set(b))
n=0
for d in range(-3,4):
    for j in range(3):
        B = Ka.basis(d,j)
        s = Ua.sigp(B); assert all((s[k]+B[k]).is_zero for k in B); n+=1
        if d < 0:
            t = Ua.tau(B); B2 = Ka.basis(-d,j); assert all((t[k]-B2[k]).is_zero for k in B2)
        # tau sigma' = sigma' tau
        x = Ua.tau(Ua.sigp(B)); y = Ua.sigp(Ua.tau(B)); assert all((x[k]-y[k]).is_zero for k in x)
# closure of brackets
for a_ in [(1,0),(2,1),(-1,2),(0,1),(-3,0)]:
    for b_ in [(-1,0),(2,0),(-2,1),(0,2)]:
        Ka.brk(a_, b_)
        n+=1
print("checks", n, "ok")
# sigma = transpose and sigma' = J x^T J are conjugate (2x2 matrices)
E=sp.Matrix([[0,1],[0,0]]); F=E.T; Hm=sp.diag(1,-1); J=sp.Matrix([[0,1],[1,0]])
sp_ = lambda x: J*x.T*J
assert sp_(E)==E and sp_(F)==F and sp_(Hm)==-Hm
g = sp.Matrix([[1,1],[sp.I,-sp.I]])/sp.sqrt(2)   # g^T g = J? 
print("g^T g =", sp.simplify(g.T*g), " g(e-f)g^-1 =", sp.simplify(g.inv()*(E-F)*g))
