# checks for the subsection of the text on projective structures
import sympy as sp
z,c=sp.symbols('z c'); F=sp.Function('F')(z); phi=sp.Function('phi'); u=sp.Function('u')
S=lambda G: sp.diff(G,z,3)/sp.diff(G,z)-sp.Rational(3,2)*(sp.diff(G,z,2)/sp.diff(G,z))**2
n=0
# 1) transformation of c d^2+u on F_{-1/2} -> F_{3/2} under x=F(z)
x=sp.Symbol('x')
psi=phi(F)*sp.diff(F,z)**sp.Rational(-1,2)
lhs=(c*sp.diff(phi(x),x,2)+u(x)*phi(x)).subs(x,F).doit()*sp.diff(F,z)**sp.Rational(3,2)
ut=u(F)*sp.diff(F,z)**2+c/2*S(F)
rhs=c*sp.diff(psi,z,2)+ut*psi
assert sp.simplify(sp.expand(lhs-rhs))==0; n+=1
# 2) stabilizer: delta_g u = g u' + 2 g' u + (c/2) g''' ; products of solutions
a=sp.Function('a')(z); g=sp.Function('g')(z)
# derive delta from 1) with F = z + eps g
eps=sp.Symbol('eps'); Fe=z+eps*g
ute=u(Fe)*sp.diff(Fe,z)**2+c/2*S(Fe)
d=sp.simplify(sp.diff(ute,eps).subs(eps,0)-0)
assert sp.simplify(d-(g*sp.diff(u(z),z)+2*sp.diff(g,z)*u(z)+c/2*sp.diff(g,z,3)))==0; n+=1
# products of two solutions of c phi''+u phi=0 annihilated by the third-order operator
p1,p2=sp.Function('p1')(z),sp.Function('p2')(z)
rule={sp.diff(p1,z,2):-u(z)*p1/c, sp.diff(p2,z,2):-u(z)*p2/c}
def red(e):
    for k in range(6):
        e=sp.expand(e.subs({sp.diff(p1,z,j):sp.diff(-u(z)*p1/c,z,j-2) for j in (5,4,3,2)}).subs({sp.diff(p2,z,j):sp.diff(-u(z)*p2/c,z,j-2) for j in (5,4,3,2)}))
    return sp.simplify(e)
for G in (p1*p1,p1*p2,p2*p2):
    e=G*sp.diff(u(z),z)+2*sp.diff(G,z)*u(z)+c/2*sp.diff(G,z,3)
    assert red(e)==0; n+=1
# 3) Wronskian constant, S(p1/p2)=2u/c
W=p1*sp.diff(p2,z)-sp.diff(p1,z)*p2
assert red(sp.diff(W,z))==0; n+=1
e=S(p1/p2)-2*u(z)/c
assert red(sp.together(e).as_numer_denom()[0])==0; n+=1
# 4) bracket of products: [p_i p_j d, p_k p_l d] stays in span (sl(2)); check [p1^2, p2^2] = 2W(p1p2)... 
br=lambda A,B: A*sp.diff(B,z)-sp.diff(A,z)*B
assert red(br(p1**2,p2**2)-2*W*p1*p2)==0; n+=1
assert red(br(p1**2,p1*p2)-W*p1**2)==0; n+=1
assert red(br(p1*p2,p2**2)-W*p2**2)==0; n+=1
print("checks passed:",n)
